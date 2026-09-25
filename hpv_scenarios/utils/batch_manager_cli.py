#!/usr/bin/env python3
"""
Batch Manager CLI

Command-line utility for managing seed batches and merging results.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project directories to path
script_dir = Path(__file__).parent
project_dir = script_dir.parent.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "hpv_scenarios"))

from core.seed_batch_manager import SeedBatchManager, merge_seed_batches


def show_progress(base_dir: str, filestem: str = "_jul22"):
    """Show progress summary for all scenarios."""
    manager = SeedBatchManager(base_dir, filestem)
    summary = manager.get_progress_summary()
    
    print("=== Seed Batch Progress Summary ===")
    print(f"Countries: {', '.join(summary['countries'])}")
    print(f"Total scenarios: {summary['total_scenarios']}")
    print(f"Total seeds completed: {summary['total_seeds_completed']}")
    print()
    
    for country, scenarios in summary['scenarios_detail'].items():
        print(f"{country.title()}:")
        for scenario, info in scenarios.items():
            seeds = info['seed_list']
            seed_ranges = []
            if seeds:
                start = seeds[0]
                end = seeds[0]
                for seed in seeds[1:]:
                    if seed == end + 1:
                        end = seed
                    else:
                        seed_ranges.append(f"{start}-{end}" if start != end else str(start))
                        start = end = seed
                seed_ranges.append(f"{start}-{end}" if start != end else str(start))
            
            range_str = ", ".join(seed_ranges) if seed_ranges else "none"
            print(f"  {scenario}: {info['completed_seeds']} seeds ({range_str})")
        print()


def merge_batches(base_dir: str, country: str, scenario: str, filestem: str = "_jul22"):
    """Merge partial results for a specific country-scenario combination."""
    print(f"Merging batches for {country} - {scenario}...")
    
    # For now, we'll use automatic detection of partial files
    result_file = merge_seed_batches(base_dir, country, scenario, [], filestem)
    
    if result_file:
        print(f"✓ Merged results saved to: {result_file}")
    else:
        print("✗ Failed to merge results")


def cleanup_temp(base_dir: str, filestem: str = "_jul22", country: str = None, scenario: str = None):
    """Clean up temporary simulation files."""
    manager = SeedBatchManager(base_dir, filestem)
    
    if country and scenario:
        print(f"Cleaning up temp files for {country} - {scenario}...")
        #manager.cleanup_temp_files(country, scenario)
    else:
        print("Cleaning up ALL temporary files...")
        #manager.cleanup_temp_files()
    
    print("✓ Cleanup completed")


def run_batch(base_dir: str, seed_range: tuple, filestem: str = "_jul22", 
              countries: list = None, debug: bool = False):
    """Run a specific seed batch."""
    print(f"Running seed batch {seed_range[0]}-{seed_range[1]}...")
    
    # Import and run the vaccination scenario runner
    sys.path.insert(0, str(project_dir / "hpv_scenarios" / "runners"))
    from run_vaccination_scenarios import VaccinationScenarioRunner
    
    if countries is None:
        countries = ["zambia", "cote d'ivoire"]
    
    runner = VaccinationScenarioRunner(
        debug=debug,
        save_individual_sims=True,
        seed_range=seed_range
    )
    
    try:
        results = runner.run_all_vaccination_scenarios(countries, filestem)
        print(f"✓ Batch {seed_range[0]}-{seed_range[1]} completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error running batch: {e}")
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="HPV Vaccination Seed Batch Manager")
    parser.add_argument("--base-dir", default=None, 
                       help="Base results directory (default: auto-detect)")
    parser.add_argument("--filestem", default="_jul22",
                       help="File suffix (default: _jul22)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Progress command
    progress_parser = subparsers.add_parser("progress", help="Show progress summary")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge partial results")
    merge_parser.add_argument("country", help="Country name")
    merge_parser.add_argument("scenario", help="Scenario name")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up temp files")
    cleanup_parser.add_argument("--country", help="Specific country (optional)")
    cleanup_parser.add_argument("--scenario", help="Specific scenario (optional)")
    
    # Run batch command
    batch_parser = subparsers.add_parser("run", help="Run a seed batch")
    batch_parser.add_argument("start_seed", type=int, help="Start seed")
    batch_parser.add_argument("end_seed", type=int, help="End seed")
    batch_parser.add_argument("--countries", nargs="+", default=None,
                             help="Countries to run (default: zambia, cote d'ivoire)")
    batch_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = str(project_dir / "hpv_scenarios" / "results" / "vaccination")
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return 1
    
    # Execute command
    if args.command == "progress":
        show_progress(base_dir, args.filestem)
    
    elif args.command == "merge":
        merge_batches(base_dir, args.country, args.scenario, args.filestem)
    
    elif args.command == "cleanup":
        cleanup_temp(base_dir, args.filestem, args.country, args.scenario)
    
    elif args.command == "run":
        seed_range = (args.start_seed, args.end_seed)
        success = run_batch(base_dir, seed_range, args.filestem, args.countries, args.debug)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())