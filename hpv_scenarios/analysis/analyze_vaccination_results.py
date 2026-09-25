"""
Analyze Vaccination Results

This script loads previously saved vaccination scenario results and generates
summary reports and detailed analyses.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
from typing import List, Dict, Any, Optional

# Add directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
hpv_scenarios_dir = os.path.dirname(script_dir)  # hpv_scenarios/
main_project_dir = os.path.dirname(hpv_scenarios_dir)  # main project directory

sys.path.insert(0, hpv_scenarios_dir)  # For core.* imports
sys.path.insert(0, main_project_dir)   # For run_sim, utils, etc.

# Original codebase imports
import utils as ut
import analyzers as an


class VaccinationResultsAnalyzer:
    """
    Analyzer for loading and processing saved vaccination scenario results.
    """
    
    def __init__(self, results_dir: str = None):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir: Directory containing saved .obj files
        """
        if results_dir is None:
            self.results_dir = os.path.join(main_project_dir, "hpv_scenarios", "results", "vaccination")
        else:
            self.results_dir = results_dir
            
        self.multisims = {}
        
    def load_all_results(self, filestem: str = "_jul21") -> Dict[str, Dict[str, Any]]:
        """
        Load all vaccination scenario results from saved files.
        
        Args:
            filestem: File suffix to match
            
        Returns:
            Dictionary organized as {country: {scenario_name: multisim}}
        """
        print(f"Loading vaccination results from {self.results_dir}")
        
        # Find all result files
        pattern = os.path.join(self.results_dir, f"*{filestem}.obj")
        result_files = glob.glob(pattern)
        
        if not result_files:
            print(f"No result files found matching pattern: {pattern}")
            return {}
            
        print(f"Found {len(result_files)} result files")
        
        # Load each file and organize by country and scenario
        multisims = {}
        
        for file_path in result_files:
            try:
                filename = os.path.basename(file_path)
                print(f"Loading {filename}...")
                
                # Parse filename: country_scenario_filestem.obj
                parts = filename.replace(filestem + ".obj", "").split("_")
                
                # Handle country names with underscores (like cote_d_ivoire)
                if len(parts) >= 2:
                    # Find where scenario name starts (look for known scenario patterns)
                    scenario_start_idx = None
                    for i, part in enumerate(parts):
                        if part in ['baseline', 'vx'] or (i > 0 and parts[i-1] in ['baseline', 'vx']):
                            scenario_start_idx = i if part in ['baseline', 'vx'] else i - 1
                            break
                    
                    if scenario_start_idx is not None:
                        country = "_".join(parts[:scenario_start_idx])
                        scenario = "_".join(parts[scenario_start_idx:])
                    else:
                        # Fallback: assume last part is scenario
                        country = "_".join(parts[:-1])
                        scenario = parts[-1]
                else:
                    print(f"Warning: Could not parse filename {filename}")
                    continue
                
                # Convert country name to readable format
                country = country.replace("_", " ").replace("cote d ivoire", "cote d'ivoire")
                
                # Load the multisim
                msim = sc.loadobj(file_path)
                
                # Initialize country dict if needed
                if country not in multisims:
                    multisims[country] = {}
                
                multisims[country][scenario] = msim
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        self.multisims = multisims
        
        # Print summary
        for country, scenarios in multisims.items():
            print(f"{country}: {len(scenarios)} scenarios loaded")
            
        return multisims
    
    def generate_summary_report(self, multisims: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate summary report of all scenario results.
        
        Args:
            multisims: Dictionary of MultiSim results (uses loaded if None)
            
        Returns:
            Summary DataFrame
        """
        if multisims is None:
            multisims = self.multisims
            
        if not multisims:
            print("No multisims available. Load results first.")
            return pd.DataFrame()
            
        summary_data = []
        
        for country, country_results in multisims.items():
            for scenario_name, msim in country_results.items():
                
                print(f"Processing {country} - {scenario_name}")
                
                # Extract key metrics for multiple years
                for final_year in [2030, 2040, 2050, 2060, 2070, 2080, 2090]:
                    # Use the year attribute instead of yearvec
                    year_idx = list(msim.results.year).index(final_year) if final_year in msim.results.year else -1
                    
                    if year_idx < 0:
                        continue  # Skip years not in results
                    
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
                                    print(f"  Year {final_year} not found in genotype analyzer data")
                            else:
                                print(f"  genotype_analyzer missing expected attributes")
                        else:
                            print(f"  genotype_analyzer not found in MultiSim")
                    except Exception as e:
                        print(f"  Error accessing genotype analyzer: {e}")
                        genotype_data = {}  # Empty data on error
                    
                    # Combine all data
                    row_data = {
                        'country': country,
                        'scenario': scenario_name,
                        'n_sims': msim.meta.n_sims if hasattr(msim.meta, 'n_sims') else 1,
                        'year': final_year,
                        'hpv_incidence': hpv_incidence,
                        'cancer_incidence': cancer_incidence,
                        'cancer_deaths': cancer_deaths,
                        'total_vaccinated': total_vaccinated
                    }
                    row_data.update(genotype_data)
                    summary_data.append(row_data)
        
        return pd.DataFrame(summary_data)
    
    def save_summary_report(self, summary_df: pd.DataFrame, 
                           output_file: str = None, filestem: str = "_jul21"):
        """
        Save summary report to CSV file.
        
        Args:
            summary_df: Summary DataFrame to save
            output_file: Output filename (auto-generated if None)
            filestem: File suffix for auto-generated filename
        """
        if output_file is None:
            output_file = os.path.join(self.results_dir, f"vaccination_scenarios_summary_analysis{filestem}.csv")
        
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary report saved: {output_file}")
        print(f"Summary contains {len(summary_df)} rows and {len(summary_df.columns)} columns")
        
        # Show column names
        print(f"Columns: {list(summary_df.columns)}")
        
        # Show first few rows
        if len(summary_df) > 0:
            print("\nFirst 5 rows:")
            print(summary_df.head())
        
        return output_file
    
    def analyze_genotype_patterns(self, summary_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analyze genotype-specific patterns across scenarios.
        
        Args:
            summary_df: Summary DataFrame (generates if None)
            
        Returns:
            DataFrame with genotype-specific analysis
        """
        if summary_df is None:
            summary_df = self.generate_summary_report()
        
        if summary_df.empty:
            print("No data available for genotype analysis")
            return pd.DataFrame()
        
        # Filter for genotype columns
        genotype_cols = [col for col in summary_df.columns 
                        if any(gtype in col for gtype in ['hpv16', 'hpv18', 'hi5', 'ohr'])]
        
        if not genotype_cols:
            print("No genotype-specific data found in summary")
            return pd.DataFrame()
        
        print(f"Found genotype-specific columns: {genotype_cols}")
        
        # Create genotype-focused analysis
        base_cols = ['country', 'scenario', 'year']
        analysis_df = summary_df[base_cols + genotype_cols].copy()
        
        return analysis_df


def main():
    """Main analysis function."""
    
    print("HPV Vaccination Results Analysis")
    print("=" * 50)
    
    # Configuration
    filestem = "_jul21"
    
    # Initialize analyzer
    analyzer = VaccinationResultsAnalyzer()
    
    # Load all results
    multisims = analyzer.load_all_results(filestem=filestem)
    
    if not multisims:
        print("No results loaded. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("Generating Summary Report")
    print("=" * 50)
    
    # Generate summary report
    summary_df = analyzer.generate_summary_report(multisims)
    
    if summary_df.empty:
        print("No summary data generated.")
        return
    
    # Save summary report
    output_file = analyzer.save_summary_report(summary_df, filestem=filestem)
    
    # Generate genotype analysis if available
    print("\n" + "=" * 50)
    print("Analyzing Genotype Patterns")
    print("=" * 50)
    
    genotype_df = analyzer.analyze_genotype_patterns(summary_df)
    
    if not genotype_df.empty:
        genotype_file = os.path.join(analyzer.results_dir, f"genotype_analysis{filestem}.csv")
        genotype_df.to_csv(genotype_file, index=False)
        print(f"Genotype analysis saved: {genotype_file}")
    
    print("\n" + "=" * 50)
    print("Analysis completed successfully!")
    print(f"Summary report: {output_file}")
    if not genotype_df.empty:
        print(f"Genotype analysis: {genotype_file}")


if __name__ == "__main__":
    main()