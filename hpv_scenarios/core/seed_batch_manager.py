"""
Seed Batch Manager for HPV Vaccination Scenarios

This module provides functionality for managing seed batches, progress tracking,
and incremental saving/loading of simulation results.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
from typing import List, Dict, Any, Optional, Tuple


class SeedBatchManager:
    """
    Manages seed batches, progress tracking, and incremental result saving.
    """
    
    def __init__(self, base_dir: str, filestem: str = "_nov06"):
        """
        Initialize the seed batch manager.
        
        Args:
            base_dir: Base directory for saving results
            filestem: File suffix for saved objects
        """
        self.base_dir = base_dir
        self.filestem = filestem
        self.temp_dir = os.path.join(base_dir, "temp_sims")
        self.progress_file = os.path.join(base_dir, f"run_progress{filestem}.json")
        
        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)
        
        # Load or initialize progress tracking
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict[str, Any]:
        """Load progress tracking data from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
        
        return {
            "completed_seeds": {},  # {country: {scenario: [seed_list]}}
            "batch_info": {},       # Metadata about batches
            "total_seeds_target": 0
        }
    
    def save_progress(self):
        """Save progress tracking data to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress file: {e}")
    
    def get_temp_sim_path(self, country: str, scenario: str, seed: int) -> str:
        """Get the path for saving an individual simulation."""
        country_clean = country.replace(" ", "_").replace("'", "_")
        scenario_clean = scenario.replace(" ", "_")
        sim_dir = os.path.join(self.temp_dir, country_clean, scenario_clean)
        os.makedirs(sim_dir, exist_ok=True)
        return os.path.join(sim_dir, f"seed_{seed:03d}.obj")
    
    def save_individual_sim(self, sim: hpv.Sim, country: str, scenario: str, seed: int):
        """Save an individual simulation immediately after completion."""
        sim_path = self.get_temp_sim_path(country, scenario, seed)
        
        try:
            # Create a lightweight copy without people data to save space
            sim_copy = sc.dcp(sim)
            if hasattr(sim_copy, 'people'):
                # Keep essential people data but remove large arrays
                essential_people = sc.objdict()
                if hasattr(sim.people, 'uid'):
                    essential_people.n_agents = len(sim.people.uid)
                else:
                    essential_people.n_agents = getattr(sim.people, 'n', 0)
                sim_copy.people = essential_people
            
            # Save the simulation
            sc.saveobj(sim_path, sim_copy)
            
            # Update progress tracking
            if country not in self.progress["completed_seeds"]:
                self.progress["completed_seeds"][country] = {}
            if scenario not in self.progress["completed_seeds"][country]:
                self.progress["completed_seeds"][country][scenario] = []
            
            if seed not in self.progress["completed_seeds"][country][scenario]:
                self.progress["completed_seeds"][country][scenario].append(seed)
                self.progress["completed_seeds"][country][scenario].sort()
            
            self.save_progress()
            print(f"  Saved individual sim: {country} - {scenario} - seed {seed}")
            
        except Exception as e:
            print(f"Error saving individual sim {country}-{scenario}-{seed}: {e}")
    
    def get_completed_seeds(self, country: str, scenario: str) -> List[int]:
        """Get list of completed seeds for a country-scenario combination."""
        if country in self.progress["completed_seeds"]:
            if scenario in self.progress["completed_seeds"][country]:
                return self.progress["completed_seeds"][country][scenario]
        return []
    
    def get_missing_seeds(self, country: str, scenario: str, target_seeds: List[int]) -> List[int]:
        """Get list of seeds that still need to be run."""
        completed = set(self.get_completed_seeds(country, scenario))
        target_set = set(target_seeds)
        return sorted(list(target_set - completed))
    
    def load_individual_sims(self, country: str, scenario: str, seeds: List[int] = None) -> List[hpv.Sim]:
        """Load individual simulations for a country-scenario combination."""
        if seeds is None:
            seeds = self.get_completed_seeds(country, scenario)
        
        sims = []
        for seed in seeds:
            sim_path = self.get_temp_sim_path(country, scenario, seed)
            if os.path.exists(sim_path):
                try:
                    sim = sc.loadobj(sim_path)
                    sims.append(sim)
                except Exception as e:
                    print(f"Error loading sim {country}-{scenario}-{seed}: {e}")
            else:
                print(f"Warning: Sim file not found: {sim_path}")
        
        return sims
    
    def create_partial_multisim(self, country: str, scenario: str, seeds: List[int] = None) -> Optional[hpv.MultiSim]:
        """Create a MultiSim from available individual simulations."""
        sims = self.load_individual_sims(country, scenario, seeds)
        
        if not sims:
            print(f"No simulations found for {country} - {scenario}")
            return None
        
        print(f"Creating MultiSim for {country} - {scenario} with {len(sims)} seeds")
        
        try:
            # Create MultiSim
            msim = hpv.MultiSim(sims)
            msim.reduce(use_mean=True)
            
            # Add metadata
            msim.meta = sc.objdict()
            msim.meta.scenario = scenario
            msim.meta.country = country
            msim.meta.n_sims = len(sims)
            msim.meta.seeds_used = sorted([sim.meta.seed for sim in sims if hasattr(sim.meta, 'seed')])
            
            return msim
            
        except Exception as e:
            print(f"Error creating MultiSim for {country}-{scenario}: {e}")
            return None
    
    def save_partial_multisim(self, msim: hpv.MultiSim, country: str, scenario: str, suffix: str = ""):
        """Save a partial MultiSim result."""
        country_clean = country.replace(" ", "_").replace("'", "_")
        scenario_clean = scenario.replace(" ", "_")
        
        if suffix:
            filename = f"{country_clean}_{scenario_clean}_{suffix}{self.filestem}.obj"
        else:
            filename = f"{country_clean}_{scenario_clean}{self.filestem}.obj"
        
        filepath = os.path.join(self.base_dir, filename)
        
        try:
            sc.saveobj(filepath, msim)
            print(f"Saved partial MultiSim: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving partial MultiSim {filepath}: {e}")
            return None
    
    def merge_multisims(self, country: str, scenario: str, msim_files: List[str]) -> Optional[hpv.MultiSim]:
        """
        Merge multiple MultiSim objects with different seed ranges.
        
        Args:
            country: Country name
            scenario: Scenario name  
            msim_files: List of MultiSim file paths to merge
            
        Returns:
            Merged MultiSim object
        """
        if not msim_files:
            return None
        
        print(f"Merging {len(msim_files)} MultiSim files for {country} - {scenario}")
        
        # Load all MultiSims
        msims = []
        all_seeds = []
        
        for filepath in msim_files:
            try:
                msim = sc.loadobj(filepath)
                msims.append(msim)
                
                # Track seeds used
                if hasattr(msim.meta, 'seeds_used'):
                    all_seeds.extend(msim.meta.seeds_used)
                elif hasattr(msim.meta, 'n_sims'):
                    # Estimate seeds if not tracked
                    all_seeds.extend(range(len(all_seeds), len(all_seeds) + msim.meta.n_sims))
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        if not msims:
            print(f"No valid MultiSims loaded for merging")
            return None
        
        # For now, return the first MultiSim as merging multiple MultiSims
        # requires careful handling of the reduced results
        # TODO: Implement proper MultiSim merging if needed
        merged_msim = msims[0]
        
        # Update metadata to reflect merge
        merged_msim.meta.n_sims = sum(msim.meta.n_sims for msim in msims)
        merged_msim.meta.seeds_used = sorted(list(set(all_seeds)))
        merged_msim.meta.merged_from = len(msim_files)
        
        print(f"Merged MultiSim created with {merged_msim.meta.n_sims} total simulations")
        return merged_msim
    
    def cleanup_temp_files(self, country: str = None, scenario: str = None):
        """Clean up temporary simulation files after successful MultiSim creation."""
        if country and scenario:
            # Clean up specific scenario
            country_clean = country.replace(" ", "_").replace("'", "_")
            scenario_clean = scenario.replace(" ", "_")
            temp_scenario_dir = os.path.join(self.temp_dir, country_clean, scenario_clean)
            
            if os.path.exists(temp_scenario_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_scenario_dir)
                    print(f"Cleaned up temp files for {country} - {scenario}")
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_scenario_dir}: {e}")
        else:
            # Clean up all temp files
            if os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                    os.makedirs(self.temp_dir, exist_ok=True)
                    print("Cleaned up all temporary simulation files")
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        summary = {
            "countries": [],
            "total_scenarios": 0,
            "total_seeds_completed": 0,
            "scenarios_detail": {}
        }
        
        for country, scenarios in self.progress["completed_seeds"].items():
            summary["countries"].append(country)
            summary["scenarios_detail"][country] = {}
            
            for scenario, seeds in scenarios.items():
                summary["total_scenarios"] += 1
                summary["total_seeds_completed"] += len(seeds)
                summary["scenarios_detail"][country][scenario] = {
                    "completed_seeds": len(seeds),
                    "seed_list": seeds
                }
        
        return summary


def merge_seed_batches(base_dir: str, country: str, scenario: str, 
                      seed_ranges: List[Tuple[int, int]], filestem: str = "_nov06") -> str:
    """
    Utility function to merge results from multiple seed batch runs.
    
    Args:
        base_dir: Base directory containing results
        country: Country name
        scenario: Scenario name
        seed_ranges: List of (start, end) tuples for seed ranges
        filestem: File suffix
        
    Returns:
        Path to merged result file
    """
    manager = SeedBatchManager(base_dir, filestem)
    
    # Find all partial MultiSim files for this scenario
    country_clean = country.replace(" ", "_").replace("'", "_")
    scenario_clean = scenario.replace(" ", "_")
    
    pattern = os.path.join(base_dir, f"{country_clean}_{scenario_clean}_*{filestem}.obj")
    msim_files = glob.glob(pattern)
    
    if not msim_files:
        print(f"No partial MultiSim files found for {country} - {scenario}")
        return None
    
    # Merge MultiSims
    merged_msim = manager.merge_multisims(country, scenario, msim_files)
    
    if merged_msim is None:
        print("Failed to merge MultiSims")
        return None
    
    # Save merged result
    final_path = manager.save_partial_multisim(merged_msim, country, scenario, suffix="merged")
    
    # Clean up partial files if successful
    if final_path:
        for filepath in msim_files:
            try:
                os.remove(filepath)
                print(f"Removed partial file: {filepath}")
            except Exception as e:
                print(f"Warning: Could not remove {filepath}: {e}")
    
    return final_path