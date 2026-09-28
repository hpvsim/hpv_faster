"""
Time series plotting for HPV vaccination scenarios.

This module provides functions to plot HPV infections and cancer cases over time
for the 30 years following vaccination introduction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sciris as sc
from typing import Dict, List, Optional, Tuple
import os

def load_scenario_results(results_dir: str = "../results/vaccination", 
                         filestem: str = "_nov06") -> Dict[str, Dict[str, sc.objdict]]:
    """
    Load all vaccination scenario results.
    
    Args:
        results_dir: Directory containing result files
        filestem: File suffix for results
        
    Returns:
        Dictionary organized by country and scenario name
    """
    results = {}
    countries = ["zambia", "cote_d_ivoire", "sierra_leone"]
    scenarios = [
        "baseline_national", "vx_annual_9-10_NAT", "vx_biennial_9-11_NAT", 
        "vx_triennial_9-12_NAT", "vx_quadrennial_9-13_NAT", "vx_quinquennial_9-14_NAT",
        "baseline_high_risk", "vx_annual_9-10_HR", "vx_biennial_9-11_HR",
        "vx_triennial_9-12_HR", "vx_quadrennial_9-13_HR", "vx_quinquennial_9-14_HR"
    ]
    
    for country in countries:
        results[country] = {}
        for scenario in scenarios:
            filepath = f"{results_dir}/{country}_{scenario}{filestem}.obj"
            if os.path.exists(filepath):
                try:
                    results[country][scenario] = sc.loadobj(filepath)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
            else:
                print(f"Warning: File not found: {filepath}")
    
    return results

def extract_time_series(msim: sc.objdict, start_year: int = 2025, 
                       end_year: int = 2055) -> pd.DataFrame:
    """
    Extract time series data from a MultiSim result.
    
    Args:
        msim: MultiSim result object
        start_year: Start year for extraction
        end_year: End year for extraction
        
    Returns:
        DataFrame with time series data
    """
    years = msim.results.year
    start_idx = list(years).index(float(start_year)) if start_year in years else 0
    end_idx = list(years).index(float(end_year)) if end_year in years else len(years)
    
    data = {
        'year': years[start_idx:end_idx],
        'hpv_incidence': msim.results.hpv_incidence[start_idx:end_idx],
        'hpv_infections': msim.results.infections[start_idx:end_idx],
        'cancer_incidence': msim.results.cancer_incidence[start_idx:end_idx], 
        'cancer_cases': msim.results.cancers[start_idx:end_idx],
        'cancer_deaths': msim.results.cancer_deaths[start_idx:end_idx],
        'population': msim.results.n_alive[start_idx:end_idx] if hasattr(msim.results, 'n_alive') else None
    }
    
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}
    
    return pd.DataFrame(data)

def plot_hpv_time_series(results: Dict[str, Dict[str, sc.objdict]], 
                        country: str = "zambia",
                        risk_level: str = "national",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot HPV infections and incidence over time for different vaccination scenarios.
    
    Args:
        results: Dictionary of loaded results
        country: Country to plot
        risk_level: "national" or "high_risk"
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    # Filter scenarios by risk level
    suffix = "NAT" if risk_level == "national" else "HR"
    baseline_name = f"baseline_{risk_level}"
    
    scenario_names = [baseline_name]
    vx_scenarios = [s for s in results[country].keys() if s.endswith(suffix) and "vx_" in s]
    scenario_names.extend(sorted(vx_scenarios))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(scenario_names)))
    
    for i, scenario in enumerate(scenario_names):
        if scenario not in results[country]:
            continue
            
        # Extract time series
        ts_data = extract_time_series(results[country][scenario])
        
        # Plot HPV infections
        ax1.plot(ts_data['year'], ts_data['hpv_infections'], 
                color=colors[i], linewidth=2, label=scenario.replace('_', ' ').title())
        
        # Plot HPV incidence
        ax2.plot(ts_data['year'], ts_data['hpv_incidence'] * 1000,  # Convert to per 1000
                color=colors[i], linewidth=2, label=scenario.replace('_', ' ').title())
    
    # Format plots
    ax1.set_title(f'HPV Infections Over Time - {country.title()} ({risk_level.title()})')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('New HPV Infections')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f'HPV Incidence Rate Over Time - {country.title()} ({risk_level.title()})')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('HPV Incidence (per 1,000 person-years)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_cancer_time_series(results: Dict[str, Dict[str, sc.objdict]], 
                           country: str = "zambia",
                           risk_level: str = "national",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cancer cases and incidence over time for different vaccination scenarios.
    
    Args:
        results: Dictionary of loaded results
        country: Country to plot
        risk_level: "national" or "high_risk" 
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    # Filter scenarios by risk level
    suffix = "NAT" if risk_level == "national" else "HR"
    baseline_name = f"baseline_{risk_level}"
    
    scenario_names = [baseline_name]
    vx_scenarios = [s for s in results[country].keys() if s.endswith(suffix) and "vx_" in s]
    scenario_names.extend(sorted(vx_scenarios))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(scenario_names)))
    
    for i, scenario in enumerate(scenario_names):
        if scenario not in results[country]:
            continue
            
        # Extract time series
        ts_data = extract_time_series(results[country][scenario])
        
        # Plot cancer cases
        ax1.plot(ts_data['year'], ts_data['cancer_cases'], 
                color=colors[i], linewidth=2, label=scenario.replace('_', ' ').title())
        
        # Plot cancer incidence
        ax2.plot(ts_data['year'], ts_data['cancer_incidence'],
                color=colors[i], linewidth=2, label=scenario.replace('_', ' ').title())
    
    # Format plots
    ax1.set_title(f'Cancer Cases Over Time - {country.title()} ({risk_level.title()})')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('New Cancer Cases')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f'Cancer Incidence Rate Over Time - {country.title()} ({risk_level.title()})')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cancer Incidence Rate')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_all_plots(results_dir: str = "../results/vaccination",
                    output_dir: str = "../results/plots",
                    filestem: str = "_nov06"):
    """
    Create all time series plots for both countries and risk levels.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save plots
        filestem: File suffix for results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    print("Loading vaccination scenario results...")
    results = load_scenario_results(results_dir, filestem)
    
    countries = ["zambia", "cote_d_ivoire", "sierra_leone"]
    risk_levels = ["national", "high_risk"]
    
    for country in countries:
        for risk_level in risk_levels:
            print(f"Creating plots for {country} - {risk_level}")
            
            # HPV plots
            fig_hpv = plot_hpv_time_series(results, country, risk_level)
            hpv_filename = f"{output_dir}/{country}_{risk_level}_hpv_time_series.png"
            fig_hpv.savefig(hpv_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_hpv)
            print(f"  Saved: {hpv_filename}")
            
            # Cancer plots  
            fig_cancer = plot_cancer_time_series(results, country, risk_level)
            cancer_filename = f"{output_dir}/{country}_{risk_level}_cancer_time_series.png"
            fig_cancer.savefig(cancer_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_cancer)
            print(f"  Saved: {cancer_filename}")
    
    print("All plots completed!")

if __name__ == "__main__":
    # Create all plots
    create_all_plots()