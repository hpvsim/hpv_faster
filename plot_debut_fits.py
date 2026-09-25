#!/usr/bin/env python3
"""
Plot script to visualize sexual debut data and fitted distributions.
Compares normal and lognormal fits on the same figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm
from read_sbdata import read_debut_data
import utils as ut

def plot_debut_comparison():
    """
    Create plots comparing normal and lognormal fits to the sexual debut data
    """
    
    # Set up plotting style
    plt.style.use('default')
    ut.set_font(12)
    
    # Read data and fitted parameters for both distributions
    print("Loading lognormal fits...")
    countries_ln, dff_ln, df2_ln, rvs_ln = read_debut_data(dist_type='lognormal')
    print("Loading normal fits...")  
    countries_n, dff_n, df2_n, rvs_n = read_debut_data(dist_type='normal')
    
    # Create figure with subplots for each country and sex
    countries = countries_ln
    n_countries = len(countries)
    
    for sex_key, sex_name in {'f': 'Women', 'm': 'Men'}.items():
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        axes = axes.flatten()
        
        dfw_ln = dff_ln[sex_name]
        dfw_n = dff_n[sex_name]
        
        for pn, country in enumerate(countries):
            if pn >= len(axes):
                break
            if pd.isna(country):
                continue
            ax = axes[pn]
            
            # Map country name for data lookup
            print(country)
            data_country = ut.map_sb_loc(country)
            
            # Plot the actual data points
            dfplot = dfw_ln.loc[(dfw_ln["Country"] == data_country)]
            if len(dfplot) > 0:
                # Remove any artificial age 60 entries for cleaner plotting
                dfplot_clean = dfplot[dfplot["Age"] <= 25]
                sns.scatterplot(ax=ax, data=dfplot_clean, x="Age", y="Percentage", 
                              color='black', s=60, alpha=0.8, label='Data')
            
            # Plot median from DHS data
            dfmed = df2_ln.loc[df2_ln["Country"] == data_country]
            if len(dfmed) > 0:
                median_age = dfmed[f"{sex_name} median"].iloc[0]
                ax.axvline(x=median_age, color='red', linestyle=':', alpha=0.7, 
                          label=f'DHS Median ({median_age:.1f})')
            
            # Plot fitted distributions
            xx = np.arange(8, 25, 0.1)
            
            # Lognormal fit
            if data_country in rvs_ln[sex_name]:
                rv_ln = rvs_ln[sex_name][data_country]
                ax.plot(xx, rv_ln.cdf(xx) * 100, 'b-', lw=2, 
                       label=f'Lognormal (μ={rv_ln.mean():.1f}, σ={rv_ln.std():.1f})')
            
            # Normal fit  
            if data_country in rvs_n[sex_name]:
                rv_n = rvs_n[sex_name][data_country]
                ax.plot(xx, rv_n.cdf(xx) * 100, 'g--', lw=2,
                       label=f'Normal (μ={rv_n.mean():.1f}, σ={rv_n.std():.1f})')
            
            # Format subplot
            title_country = country.title()
            if title_country == 'Congo Democratic Republic':
                title_country = 'DRC'
            
            ax.set_title(f'{title_country} - {sex_name}')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Cumulative % sexually active')
            ax.set_xlim(8, 22)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Remove empty subplots
        #for i in range(len(countries), len(axes)):
         #   fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(f"figures/debut_fits_{sex_name.lower()}.png", dpi=300, bbox_inches='tight')
        print(f"Saved plot: figures/debut_fits_{sex_name.lower()}.png")
    
    plt.show()

def create_summary_table():
    """
    Create a summary table of fitted parameters
    """
    
    # Load parameter files
    women_ln = pd.read_csv('data/sb_pars_women_lognormal.csv')
    men_ln = pd.read_csv('data/sb_pars_men_lognormal.csv') 
    women_n = pd.read_csv('data/sb_pars_women_normal.csv')
    men_n = pd.read_csv('data/sb_pars_men_normal.csv')
    
    print("\n" + "="*80)
    print("SEXUAL DEBUT DISTRIBUTION PARAMETERS SUMMARY")
    print("="*80)
    
    print("\nLOGNORMAL DISTRIBUTION PARAMETERS:")
    print("-" * 50)
    print(f"{'Country':<25} {'Women μ':<10} {'Women σ':<10} {'Men μ':<10} {'Men σ':<10}")
    print("-" * 50)
    
    for i, country in enumerate(women_ln['location']):
        w_mu = women_ln.iloc[i]['par1']
        w_sig = women_ln.iloc[i]['par2']
        m_mu = men_ln.iloc[i]['par1'] 
        m_sig = men_ln.iloc[i]['par2']
        print(f"{country:<25} {w_mu:<10.2f} {w_sig:<10.2f} {m_mu:<10.2f} {m_sig:<10.2f}")
    
    print("\nNORMAL DISTRIBUTION PARAMETERS:")
    print("-" * 50)
    print(f"{'Country':<25} {'Women μ':<10} {'Women σ':<10} {'Men μ':<10} {'Men σ':<10}")
    print("-" * 50)
    
    for i, country in enumerate(women_n['location']):
        w_mu = women_n.iloc[i]['par1']
        w_sig = women_n.iloc[i]['par2']
        m_mu = men_n.iloc[i]['par1']
        m_sig = men_n.iloc[i]['par2']
        print(f"{country:<25} {w_mu:<10.2f} {w_sig:<10.2f} {m_mu:<10.2f} {m_sig:<10.2f}")

if __name__ == '__main__':
    
    print("Creating sexual debut fit comparison plots...")
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Generate plots
    plot_debut_comparison()
    
    # Print summary table
    create_summary_table()
    
    print("\nPlotting complete!")