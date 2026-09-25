"""
HPV Vaccination Scenario Runner

This script runs vaccination scenarios with different schedules and risk profiles
for Cote d'Ivoire and Zambia using the modular scenario framework.
"""

import os
import sys

# Add directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
hpv_scenarios_dir = os.path.dirname(script_dir)  # hpv_scenarios/
main_project_dir = os.path.dirname(hpv_scenarios_dir)  # main project directory

sys.path.insert(0, hpv_scenarios_dir)  # For core.* imports
sys.path.insert(0, main_project_dir)   # For run_sim, utils, etc.

# Configure numpy threading for cluster environments
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1", 
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
import threading
import platform

# Cross-platform file locking
if platform.system() == 'Windows':
    import msvcrt
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl
    def lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# Framework imports
from core.scenario_config import ScenarioBuilder, ScenarioDefinition, ScenarioValidator
from core.risk_profiles import RiskProfileManager

# Original codebase imports
import run_sim as rs
import utils as ut
import analyzers as an
import pars_data as dp


class IncrementalSaver:
    """Manages incremental saving of simulation results to avoid memory issues and hanging."""
    
    def __init__(self, output_dir: str, filestem: str = "_nov06"):
        self.output_dir = Path(output_dir)
        self.filestem = filestem
        self.temp_dir = self.output_dir / "temp_sims"
        self.progress_file = self.output_dir / f"progress{filestem}.json"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety for progress file
        self.progress_lock = threading.Lock()
        
        # Track completion status
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load existing progress from checkpoint file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
        
        return {
            "completed_sims": {},  # {scenario_country_seed: filepath}
            "completed_multisims": {},  # {scenario_country: filepath}
            "start_time": time.time(),
            "last_update": time.time()
        }
    
    def save_progress(self):
        """Save current progress to checkpoint file with thread safety."""
        self.progress["last_update"] = time.time()
        
        with self.progress_lock:  # Thread-safe access
            try:
                # Use file locking for additional protection against race conditions
                with open(self.progress_file, 'a+') as f:
                    # Lock the file for exclusive access
                    lock_file(f)
                    
                    try:
                        # Read existing progress
                        f.seek(0)
                        content = f.read()
                        
                        if content.strip():
                            existing = json.loads(content)
                            # Merge completed_sims and completed_multisims
                            for key in ["completed_sims", "completed_multisims"]:
                                if key in existing and isinstance(existing[key], dict):
                                    self.progress[key].update(existing[key])
                        
                        # Write updated progress
                        f.seek(0)
                        f.truncate()
                        json.dump(self.progress, f, indent=2)
                        f.flush()
                        
                    finally:
                        # Release the file lock
                        unlock_file(f)
                        
            except Exception as e:
                print(f"Warning: Could not save progress file: {e}")
    
    def save_individual_sim(self, sim, scenario_name: str, country: str, seed: int) -> str:
        """Save individual simulation result and return filepath."""
        country_clean = country.lower().replace(" ", "_").replace("'", "_")
        filename = f"{country_clean}_{scenario_name}_seed{seed}{self.filestem}.obj"
        filepath = self.temp_dir / filename
        
        try:
            # Save the simulation
            sim.shrink()
            sc.saveobj(str(filepath), sim)
            
            # Track in progress
            sim_key = f"{scenario_name}_{country}_{seed}"
            self.progress["completed_sims"][sim_key] = str(filepath)
            self.save_progress()
            
            print(f"  ✓ Saved individual sim: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"  ✗ Error saving individual sim {filename}: {e}")
            return None
    
    def try_create_multisim(self, scenarios: List, countries: List, scenario_idx: int, country_idx: int, n_seeds: int) -> bool:
        """Try to create and save MultiSim if all seeds are complete."""
        scenario = scenarios[scenario_idx]
        # Get country from the scenario's risk profile, not from countries list
        country = scenario.risk_profile.country if scenario.risk_profile else countries[0]
        
        # Check if all seeds are complete
        seed_sims = []
        for seed in range(n_seeds):
            sim_key = f"{scenario.name}_{country}_{seed}"
            if sim_key in self.progress["completed_sims"]:
                filepath = self.progress["completed_sims"][sim_key]
                if Path(filepath).exists():
                    try:
                        sim = sc.loadobj(filepath)
                        seed_sims.append(sim)
                    except Exception as e:
                        print(f"  ✗ Error loading seed sim {filepath}: {e}")
                        return False
                else:
                    return False  # File missing
            else:
                return False  # Seed not complete
        
        if len(seed_sims) < n_seeds:
            return False  # Not all seeds ready
        
        # Create MultiSim
        try:
            print(f"  Creating MultiSim for {scenario.name} - {country} ({len(seed_sims)} seeds)")
            msim = make_msims(seed_sims, use_mean=True)
            
            # Save MultiSim
            country_clean = country.lower().replace(" ", "_").replace("'", "_")
            filename = f"{country_clean}_{scenario.name}{self.filestem}.obj"
            filepath = self.output_dir / filename
            msim.shrink()
            sc.saveobj(str(filepath), msim)
            
            # Track completion
            multisim_key = f"{scenario.name}_{country}"
            self.progress["completed_multisims"][multisim_key] = str(filepath)
            self.save_progress()
            
            print(f"  ✓ Saved MultiSim: {filename}")
            
            # Clean up individual seed files to save space
            # DISABLED: Keep temp files for debugging/analysis
            # for seed in range(n_seeds):
            #     sim_key = f"{scenario.name}_{country}_{seed}"
            #     if sim_key in self.progress["completed_sims"]:
            #         seed_filepath = self.progress["completed_sims"][sim_key]
            #         try:
            #             Path(seed_filepath).unlink(missing_ok=True)
            #             del self.progress["completed_sims"][sim_key]
            #         except Exception as e:
            #             print(f"  Warning: Could not clean up {seed_filepath}: {e}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error creating MultiSim for {scenario.name} - {country}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_sim_complete(self, scenario_name: str, country: str, seed: int) -> bool:
        """Check if a specific simulation is already complete."""
        sim_key = f"{scenario_name}_{country}_{seed}"
        return sim_key in self.progress["completed_sims"]
    
    def is_multisim_complete(self, scenario_name: str, country: str) -> bool:
        """Check if a specific MultiSim is already complete."""
        multisim_key = f"{scenario_name}_{country}"
        return multisim_key in self.progress["completed_multisims"]
    
    def get_completion_summary(self, scenarios: List, countries: List, n_seeds: int) -> Dict:
        """Get summary of completion status."""
        # Filter scenarios to only count those for the specified countries
        filtered_scenarios = [s for s in scenarios if s.risk_profile and s.risk_profile.country in countries]
        total_sims = len(filtered_scenarios) * n_seeds
        total_multisims = len(filtered_scenarios)
        
        completed_sims = len(self.progress["completed_sims"])
        completed_multisims = len(self.progress["completed_multisims"])
        
        return {
            "individual_sims": f"{completed_sims}/{total_sims}",
            "multisims": f"{completed_multisims}/{total_multisims}",
            "individual_pct": completed_sims / total_sims * 100 if total_sims > 0 else 0,
            "multisim_pct": completed_multisims / total_multisims * 100 if total_multisims > 0 else 0,
            "elapsed_time": time.time() - self.progress["start_time"]
        }


def make_msims(sims, use_mean=True):
    """
    Utility function to take a slice of sims and turn it into a multisim.
    Adapted from run_scenarios.py for proper statistical aggregation.
    """
    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    
    # Store metadata from first sim (they should all be the same except seed)
    msim.meta = sc.objdict()
    msim.meta.scenario = sims[0].meta.scenario
    msim.meta.country = sims[0].meta.country
    msim.meta.risk_level = sims[0].meta.risk_level
    msim.meta.n_sims = len(sims)
    
    # Remove seed from vals since we're aggregating across seeds
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    if 'seed' in msim.meta.vals:
        msim.meta.vals.pop("seed")
    
    print(f"Processing multisim {msim.meta.scenario} - {msim.meta.country} ({len(sims)} seeds)")
    
    return msim


class VaccinationScenarioRunner:
    """Main runner for vaccination scenario analysis."""
    
    def __init__(self, debug: bool = False, data_path: str = None):
        """
        Initialize the scenario runner.
        
        Args:
            debug: If True, run with smaller parameters for testing
            data_path: Path to data directory
        """
        self.debug = debug
        
        # Set data path relative to main project directory
        if data_path is None:
            self.data_path = os.path.join(main_project_dir, "data")
        else:
            self.data_path = data_path
        
        # Initialize components
        self.scenario_builder = ScenarioBuilder()
        self.risk_manager = RiskProfileManager(self.data_path)
        self.validator = ScenarioValidator()
        
        # Configure run parameters based on debug mode
        self.n_seeds = 1 if debug else 5  # Always use 3 seeds for statistical robustness
        self.n_agents = 1e2 if debug else 100e3  # Increased for better epidemiology
        self.ms_agent_ratio = 10  # Dynamic weighting for cancer progression statistics
        self.n_workers = 5 if debug else 5
        self.verbose = 0.1 if debug else 0.1
        
        # Result storage
        self.results = {}
        self.economic_results = {}
    
    def create_simulation(self, scenario: ScenarioDefinition, country: str, 
                         seed: int = 0, filestem: str = "_nov06") -> hpv.Sim:
        """
        Create a single HPVsim simulation for a scenario.
        
        Args:
            scenario: Scenario definition
            country: Country name
            seed: Random seed
            filestem: Parameter file suffix
            
        Returns:
            Configured HPVsim simulation
        """
        # Change to main project directory temporarily for file access
        original_cwd = os.getcwd()
        os.chdir(main_project_dir)
        
        try:
            # Load calibrated parameters - support both nov06 JSON and legacy obj files
            country_clean = country.lower().replace(" ", "_").replace("'", "_")
            
            if filestem == "_nov06":
                # Load from best_params_combined.json
                import json
                try:
                    with open("best_params_combined.json", "r") as f:
                        all_best_params = json.load(f)
                    
                    if country_clean in all_best_params:
                        flat_pars = all_best_params[country_clean].copy()
                        # Remove metadata fields that aren't simulation parameters
                        flat_pars.pop("_country", None)
                        flat_pars.pop("_mismatch", None)
                        flat_pars.pop("_calibration_index", None)
                        
                        # Convert flattened parameters to nested structure expected by HPVsim
                        calib_pars = {}
                        for key, value in flat_pars.items():
                            if key.endswith("_cin_fn_k"):
                                # Convert hi5_cin_fn_k -> genotype_pars[hi5][cancer_fn][ld50]
                                genotype = key.replace("_cin_fn_k", "")
                                if "genotype_pars" not in calib_pars:
                                    calib_pars["genotype_pars"] = {}
                                if genotype not in calib_pars["genotype_pars"]:
                                    calib_pars["genotype_pars"][genotype] = {}
                                if "cancer_fn" not in calib_pars["genotype_pars"][genotype]:
                                    # Need complete cancer_fn structure with default values
                                    calib_pars["genotype_pars"][genotype]["cancer_fn"] = {
                                        "method": "cin_integral",
                                        "transform_prob": 0.002,
                                        "ld50": 20  # Default value, will be overridden
                                    }
                                calib_pars["genotype_pars"][genotype]["cancer_fn"]["ld50"] = value
                            elif key == "sev_dist_par1":
                                # Convert sev_dist_par1 -> sev_dist[par1]
                                # Need to preserve base structure with distribution type
                                if "sev_dist" not in calib_pars:
                                    calib_pars["sev_dist"] = {
                                        "dist": "normal_pos", 
                                        "par1": 1, 
                                        "par2": 0.2
                                    }
                                calib_pars["sev_dist"]["par1"] = value
                            elif key == "f_partners_c_par1":
                                # Convert f_partners_c_par1 -> f_partners[c][par1]
                                # Need to preserve base structure with distribution type
                                if "f_partners" not in calib_pars:
                                    calib_pars["f_partners"] = {
                                        "m": {"dist": "poisson1", "par1": 0.01},
                                        "c": {"dist": "poisson1", "par1": 0.2}  # Default value, will be overridden
                                    }
                                calib_pars["f_partners"]["c"]["par1"] = value
                            elif key == "m_partners_c_par1":
                                # Convert m_partners_c_par1 -> m_partners[c][par1]
                                # Need to preserve base structure with distribution type
                                if "m_partners" not in calib_pars:
                                    calib_pars["m_partners"] = {
                                        "m": {"dist": "poisson1", "par1": 0.01},
                                        "c": {"dist": "poisson1", "par1": 0.2}  # Default value, will be overridden
                                    }
                                calib_pars["m_partners"]["c"]["par1"] = value
                            else:
                                # Direct mapping for simple parameters like beta, f_cross_layer, m_cross_layer
                                calib_pars[key] = value
                        
                        print(f"Loaded and converted nov06 calibration parameters for {country_clean}")
                    else:
                        print(f"Warning: No nov06 calibration parameters found for {country_clean}")
                        calib_pars = None
                except FileNotFoundError:
                    print("Error: best_params_combined.json not found")
                    calib_pars = None
            else:
                # Load from legacy obj files
                results_path = "results"  # Now relative to main project dir
                calib_pars = sc.loadobj(f"{results_path}/{country_clean}_pars{filestem}.obj")
            
            # Get risk profile parameters
            if scenario.risk_profile:
                sb_params = self.risk_manager.load_profile_for_scenario(
                    country=scenario.risk_profile.country,
                    risk_level=scenario.risk_profile.risk_level,
                    subregion=scenario.risk_profile.subregion
                )
            else:
                # Default to national risk profile
                sb_params = self.risk_manager.load_profile_for_scenario(
                    country=country,
                    risk_level="national"
                )
            
            # Generate interventions and analyzers
            interventions = scenario.generate_all_interventions()
            analyzers = [
                an.dalys(start=2010),
                an.econ_analyzer(),
                an.genotype_analyzer(start=2010)
            ]
            my_init_hpv_dist=dp.init_genotype_dist[country]
            my_init_hpv_dist['ohr']=0.0
            myedges = np.array([0] + list(range(8, 61)) + [70, 80, 90, 100])
            mystandard_pop = np.array([0.12 , 0.154, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018,
                                        0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
                                        0.016, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012,
                                        0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012,
                                        0.012, 0.012, 0.012, 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.008,
                                        0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008,
                                        0.05 , 0.015, 0.005, 0.   ])
            # Create simulation parameters similar to make_sim
            mylayer_probs = dp.make_layer_probs(location=country, marriage_scale=1) 
            if scenario.risk_profile.risk_level == "high_risk":
                mylayer_probs['c'][1:, 2] =0.2
            elif scenario.risk_profile.risk_level == "ultra_high_risk": 
                mylayer_probs['c'][1:, 2] =0.5
            print(myedges.shape)
            print(mystandard_pop.shape)
            pars = dict(
                n_agents=int(self.n_agents),
                dt=0.25,  # Keep high resolution even in debug mode
                start=1960,  # Keep full burn-in period even in debug mode
                end=2060,
                network="default",
                genotypes=[16, 18, "hi5", "ohr"],
                location=country,
                debut=sb_params,  # Use risk profile directly
                mixing=dp.mixing[country],
                layer_probs=mylayer_probs,
                f_partners=dp.f_partners,
                m_partners=dp.m_partners,
                init_hpv_dist=my_init_hpv_dist,
                age_bin_edges=myedges,
                standard_pop = np.array([myedges, mystandard_pop]),
                init_hpv_prev={
                    "age_brackets": np.array([12, 17, 24, 34, 44, 64, 80, 150]),
                    "m": np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
                    "f": np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
                },
                ms_agent_ratio=self.ms_agent_ratio,  # Dynamic weighting for cancer progression
                verbose=self.verbose,
            )
            
            # Merge with calibrated parameters
            if calib_pars is not None:
                pars = sc.mergedicts(pars, calib_pars)
            
            # Create sim directly with all components
            sim = hpv.Sim(
                pars=pars,
                interventions=interventions,
                analyzers=analyzers,
                rand_seed=seed,
            )
            
            # Set up metadata in expected format for analyzers
            sim.meta = sc.objdict()
            sim.meta.scenario = scenario.name
            sim.meta.country = country
            sim.meta.risk_level = scenario.risk_profile.risk_level if scenario.risk_profile else "national"
            sim.meta.seed = seed
            
            # Add vals object expected by econ_analyzer
            sim.meta.vals = sc.objdict()
            sim.meta.vals.scen = scenario.name  # Required by econ_analyzer
            
            sim.label = f"{country}--{seed}"
            
            return sim
            
        finally:
            # Always return to original directory
            os.chdir(original_cwd)
    
    def run_scenario_batch(self, scenarios: List[ScenarioDefinition], 
                          countries: List[str], filestem: str = "_nov06", coverage: float = 0.8) -> Dict[str, Any]:
        """
        Run a batch of scenarios for multiple countries with incremental saving.
        Uses IncrementalSaver to save results as they complete and avoid hanging.
        
        Args:
            scenarios: List of scenario definitions
            countries: List of country names
            filestem: Parameter file suffix
            
        Returns:
            Dictionary of MultiSim results organized by country and scenario
        """
        # Initialize incremental saver
        output_dir = os.path.join(main_project_dir, "hpv_scenarios", "results", "vaccination_2026Sep_cov"+f"{coverage:.2f}".replace('.', 'p'))
        saver = IncrementalSaver(output_dir, filestem)
        
        # Show initial progress
        summary = saver.get_completion_summary(scenarios, countries, self.n_seeds)
        print(f"Starting batch run - Current progress:")
        print(f"  Individual sims: {summary['individual_sims']} ({summary['individual_pct']:.1f}%)")
        print(f"  MultiSims: {summary['multisims']} ({summary['multisim_pct']:.1f}%)")
        if summary['elapsed_time'] > 0:
            print(f"  Elapsed time: {summary['elapsed_time']/3600:.1f} hours")
        print()
        # Prepare iteration arguments, skipping already completed simulations
        ikw = []
        count = 0
        
        # Filter scenarios to only include those for the specified countries
        filtered_scenarios = []
        for scenario in scenarios:
            if scenario.risk_profile and scenario.risk_profile.country in countries:
                filtered_scenarios.append(scenario)
        
        n_sims = len(filtered_scenarios) * self.n_seeds
        
        # Create simulation metadata, skipping completed ones - scenarios already contain country info
        for i_sc, scenario in enumerate(filtered_scenarios):
            # Get the country from the scenario's risk profile
            country = scenario.risk_profile.country if scenario.risk_profile else countries[0]
            
            for i_s in range(self.n_seeds):
                count += 1
                
                # Skip if already complete
                if saver.is_sim_complete(scenario.name, country, i_s):
                    print(f"Skipping completed sim {count}/{n_sims}: {scenario.name} - {country} - seed {i_s}")
                    continue
                
                meta = sc.objdict()
                meta.inds = [i_sc, i_s]  # scenario index, seed index
                meta.count = count
                meta.scenario = scenario.name
                meta.country = country
                meta.risk_level = scenario.risk_profile.risk_level if scenario.risk_profile else "national"
                meta.seed = i_s
                meta.vals = sc.objdict(
                    scen=scenario.name,
                    seed=i_s,
                    country=country
                )
                
                ikw.append(sc.objdict(
                    scenario=scenario,
                    country=country,
                    seed=i_s,
                    filestem=filestem,
                    meta=meta,
                    count=count,
                    total_sims=n_sims,
                    scenario_idx=i_sc,
                    country_idx=0,  # Not used in this corrected version
                    saver=saver,
                    filtered_scenarios=filtered_scenarios
                ))
        
        # Run simulations
        sc.heading(f"Running {n_sims} vaccination scenario simulations with {self.n_seeds} seeds each...")
        
        def run_single_sim_with_saving(scenario, country, seed, filestem, meta, count, total_sims, scenario_idx, country_idx, saver, filtered_scenarios):
            """Wrapper function for parallel execution with incremental saving."""
            print(f"Running sim {count}/{total_sims}: {scenario.name} - {country} - seed {seed}")
            
            try:
                # Create and run simulation
                sim = self.create_simulation(scenario, country, seed, filestem)
                sim.meta.update(meta)
                print(f"  Created simulation, running...")
                sim.run()
                print(f"  Simulation completed successfully")
                
                # Save individual simulation immediately
                saved_path = saver.save_individual_sim(sim, scenario.name, country, seed)
                if saved_path:
                    # Try to create MultiSim if all seeds are ready
                    multisim_created = saver.try_create_multisim(
                        filtered_scenarios, countries, scenario_idx, country_idx, self.n_seeds
                    )
                    if multisim_created:
                        print(f"  ✓ MultiSim created and saved for {scenario.name} - {country}")
                
                # Clear simulation from memory immediately to save space
                del sim
                import gc
                gc.collect()
                
                return "success"  # Return success indicator instead of large sim object
                
            except Exception as e:
                print(f"Error in simulation {count}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        print(f"Running {len(ikw)} remaining simulations...")
        
        if len(ikw) == 0:
            print("All simulations already complete!")
        elif self.debug:
            # Run sequentially in debug mode
            results = []
            for kwargs in ikw:
                result = run_single_sim_with_saving(**kwargs)
                results.append(result)
        else:
            # Run in parallel for production
            results = sc.parallelize(
                run_single_sim_with_saving,
                iterkwargs=ikw,
                ncpus=self.n_workers
            )
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Show final progress
        final_summary = saver.get_completion_summary(scenarios, countries, self.n_seeds)
        print(f"\nFinal progress summary:")
        print(f"  Individual sims: {final_summary['individual_sims']} ({final_summary['individual_pct']:.1f}%)")
        print(f"  MultiSims: {final_summary['multisims']} ({final_summary['multisim_pct']:.1f}%)")
        print(f"  Total elapsed time: {final_summary['elapsed_time']/3600:.1f} hours")
        
        # Try to create any remaining MultiSims for completed seeds
        print("\nChecking for any remaining MultiSims to create...")
        for i_sc, scenario in enumerate(filtered_scenarios):
            country = scenario.risk_profile.country if scenario.risk_profile else countries[0]
            if not saver.is_multisim_complete(scenario.name, country):
                multisim_created = saver.try_create_multisim(
                    filtered_scenarios, countries, i_sc, 0, self.n_seeds
                )
                if multisim_created:
                    print(f"  ✓ Created final MultiSim for {scenario.name} - {country}")
        
        # Load all completed MultiSims
        print("\nLoading completed MultiSim results...")
        multisims = {}
        for country in countries:
            multisims[country] = {}
            country_clean = country.lower().replace(" ", "_").replace("'", "_")
            
            for scenario in scenarios:
                multisim_key = f"{scenario.name}_{country}"
                if multisim_key in saver.progress["completed_multisims"]:
                    filepath = saver.progress["completed_multisims"][multisim_key]
                    try:
                        msim = sc.loadobj(filepath)
                        multisims[country][scenario.name] = msim
                        print(f"  ✓ Loaded MultiSim: {scenario.name} - {country}")
                    except Exception as e:
                        print(f"  ✗ Error loading MultiSim {filepath}: {e}")
                else:
                    print(f"  ⚠ MultiSim not found: {scenario.name} - {country}")
        
        print(f"\nCompleted incremental processing. Loaded {sum(len(country_results) for country_results in multisims.values())} MultiSims.")
        return multisims
    
    def save_results(self, multisims: Dict[str, Any], output_dir: str = None,
                    filestem: str = "_nov06", coverage: float = 0.8):
        """
        Results are already saved incrementally, but this method provides compatibility.
        
        Args:
            multisims: Dictionary of MultiSim results (already saved)
            output_dir: Output directory for results (ignored - already saved)
            filestem: File suffix (ignored - already saved)
        """
        if output_dir is None:
            output_dir = os.path.join(main_project_dir, "hpv_scenarios", "results", "vaccination_2026Jun_cov"+f"{coverage:.2f}".replace('.', 'p'))
        
        total_saved = sum(len(country_results) for country_results in multisims.values())
        print(f"\nResults already saved incrementally during execution.")
        print(f"Location: {output_dir}")
        print(f"Total MultiSims: {total_saved}")
        
        # Optional: verify all files exist
        missing_files = []
        for country, country_results in multisims.items():
            for scenario_name in country_results.keys():
                country_clean = country.lower().replace(" ", "_").replace("'", "_")
                filename = f"{country_clean}_{scenario_name}{filestem}.obj"
                filepath = os.path.join(output_dir, filename)
                if not os.path.exists(filepath):
                    missing_files.append(filepath)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} expected files not found:")
            for filepath in missing_files[:5]:  # Show first 5
                print(f"  Missing: {filepath}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
        else:
            print("✓ All expected result files verified on disk")
    
    def generate_summary_report(self, multisims: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate summary report of all scenario results.
        
        Args:
            multisims: Dictionary of MultiSim results
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for country, country_results in multisims.items():
            for scenario_name, msim in country_results.items():
                # Extract key metrics from year 2050 (better timepoint with active population)
                for final_year in [2030, 2040, 2050, 2060]:
                    # Use the year attribute instead of yearvec
                    year_idx = list(msim.results.year).index(final_year) if final_year in msim.results.year else -1
                    
                    # Use incidence rates rather than raw numbers for better comparison
                    hpv_incidence = msim.results.hpv_incidence[year_idx] if hasattr(msim.results, 'hpv_incidence') else 0
                    cancer_incidence = msim.results.cancer_incidence[year_idx] if hasattr(msim.results, 'cancer_incidence') else 0
                    
                    # Cancer deaths
                    cancer_deaths = msim.results.cancer_deaths[year_idx] if hasattr(msim.results, 'cancer_deaths') else 0
                    
                    # Vaccinations (cumulative)
                    total_vaccinated = msim.results.n_vaccinated[-1] if hasattr(msim.results, 'n_vaccinated') else 0
                    
                    # Get genotype-specific data if available
                    genotype_data = {}
                    try:
                        # Check if MultiSim has get_analyzer method
                        if hasattr(msim, 'get_analyzer'):
                            genotype_analyzer = msim.get_analyzer(an.genotype_analyzer, die=False)
                        else:
                            print(f"Warning: MultiSim object doesn't have get_analyzer method")
                            genotype_analyzer = None
                        
                        if genotype_analyzer is not None:
                            # Get year index for genotype data
                            if hasattr(genotype_analyzer, 'rate_years') and hasattr(genotype_analyzer, 'rates'):
                                geno_year_idx = list(genotype_analyzer.rate_years).index(final_year) if final_year in genotype_analyzer.rate_years else -1
                                
                                if geno_year_idx >= 0:
                                    for genotype in ['hpv16', 'hpv18', 'hi5', 'ohr']:
                                        if genotype in genotype_analyzer.rates:
                                            genotype_data.update({
                                                f'{genotype}_infections_rate': genotype_analyzer.rates[genotype]['new_infections_rate'][geno_year_idx],
                                                f'{genotype}_cancers_rate': genotype_analyzer.rates[genotype]['new_cancers_rate'][geno_year_idx],
                                                f'{genotype}_cancer_deaths_rate': genotype_analyzer.rates[genotype]['cancer_deaths_rate'][geno_year_idx],
                                                f'{genotype}_prevalence_rate': genotype_analyzer.rates[genotype]['prevalence_rate'][geno_year_idx]
                                            })
                            else:
                                print(f"Warning: genotype_analyzer missing expected attributes")
                        else:
                            print(f"Warning: genotype_analyzer not found in MultiSim for {scenario_name} - {country}")
                    except Exception as e:
                        print(f"Error accessing genotype analyzer: {e}")
                        genotype_data = {}  # Empty data on error
                    
                    # Combine all data
                    row_data = {
                        'country': country,
                        'scenario': scenario_name,
                        'n_sims': msim.meta.n_sims,
                        'year': final_year,
                        'hpv_incidence': hpv_incidence,
                        'cancer_incidence': cancer_incidence,
                        'cancer_deaths': cancer_deaths,
                        'total_vaccinated': total_vaccinated
                    }
                    row_data.update(genotype_data)
                    summary_data.append(row_data)
        
        return pd.DataFrame(summary_data)
    
    def run_all_vaccination_scenarios(self, countries: Optional[List[str]] = None,
                                     filestem: str = "_nov06", coverage: float = 0.8) -> Dict[str, Any]:
        """
        Run all vaccination scenarios for specified countries.
        
        Args:
            countries: List of countries (default: ["zambia", "cote d'ivoire"])
            filestem: Parameter file suffix
            
        Returns:
            Dictionary of MultiSim results
        """
        if countries is None:
            countries = ["zambia", "cote d'ivoire", "sierra leone"]

        # Generate all scenarios
        self.scenario_builder.base_config['coverage'] = coverage
        scenarios = self.scenario_builder.create_vaccination_scenarios()
        
        # Validate scenarios
        print("Validating scenarios...")
        valid_scenarios = []
        for scenario in scenarios:
            issues = self.validator.validate_scenario(scenario)
            if issues:
                print(f"Warning: Issues with scenario {scenario.name}: {issues}")
            else:
                valid_scenarios.append(scenario)
        
        print(f"Running {len(valid_scenarios)} valid scenarios for {len(countries)} countries")
        
        # Run simulations with proper MultiSim structure
        multisims = self.run_scenario_batch(valid_scenarios, countries, filestem, coverage)
        
        # Save results
        self.save_results(multisims, filestem=filestem, coverage=coverage)
        
        # Generate summary
        summary = self.generate_summary_report(multisims)
        summary_file = os.path.join(main_project_dir, "hpv_scenarios", "results", "vaccination", f"vaccination_scenarios_summary{filestem}.csv")
        summary.to_csv(summary_file, index=False)
        print(f"\nSummary report saved: {summary_file}")
        print("\nSummary (first 10 rows):")
        print(summary.head(10))
        
        return multisims


def main():
    """Main execution function."""
    
    # Configuration
    debug = False  # Set to False for production runs
    filestem = "_nov06"  # Use nov06 calibration parameters by default

    # Per-country coverage grids: central value ± {1,2,3,4,5,7,10} pct for Zambia,
    # ± {1,2,3,4,5,7,9} pct for Sierra Leone (capped to avoid exceeding 1.0)
    zambia_central = 0.6165
    sl_central = 0.91
    shared_offsets   = [-0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    sl_offsets       = [-0.09, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09]

    country_coverages = {
        "zambia":       [round(zambia_central + d, 4) for d in shared_offsets],
        "sierra leone": [round(sl_central     + d, 4) for d in sl_offsets],
    }

    print("HPV Vaccination Scenario Analysis")
    print("=" * 50)
    print(f"Debug mode: {debug}")
    print(f"Countries: {list(country_coverages.keys())}")
    print(f"Parameter files: {filestem}")
    print()
    for country, coverages in country_coverages.items():
        print(f"\n--- {country.title()} ({len(coverages)} coverage values) ---")
        for coverage in coverages:
            # Initialize runner
            runner = VaccinationScenarioRunner(debug=debug)

            # Run all scenarios
            try:
                results = runner.run_all_vaccination_scenarios([country], filestem, coverage)

                print("\n" + "=" * 50)
                print("Vaccination scenario analysis completed successfully!")
                print(f"Results saved for {len(results)} countries")

                # Print final summary
                total_scenarios = sum(len(country_results) for country_results in results.values())
                print(f"Total scenario combinations: {total_scenarios}")

                # Clean up memory and ensure processes are terminated
                del results
                del runner
                import gc
                gc.collect()

            except Exception as e:
                print(f"Error running vaccination scenarios: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()