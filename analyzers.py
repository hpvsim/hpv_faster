"""
Define custom analyzers for HPVsim
"""

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import math


class dalys(hpv.Analyzer):
    """
    Analyzer for computing DALYs.
    """

    def __init__(self, start=None, life_expectancy=84, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.si = None  # Start index - calculated upon initialization based on sim time vector
        self.df = None  # Results dataframe
        self.disability_weights = sc.objdict(
            weights=[
                0.288,
                0.049,
                0.451,
                0.54,
            ],  # From GBD2017 - see Table A2.1 https://www.thelancet.com/cms/10.1016/S2214-109X(20)30022-X/attachment/0f63cf98-5eb9-48eb-af4f-6abe8fdff544/mmc1.pdf
            time_fraction=[0.05, 0.85, 0.09, 0.01],  # Estimates based on durations
        )
        self.life_expectancy = (
            life_expectancy  # Should typically use country-specific values
        )
        return

    @property
    def av_disutility(self):
        """The average disability weight over duration of cancer"""
        dw = self.disability_weights
        len_dw = len(dw.weights)
        return sum([dw.weights[i] * dw.time_fraction[i] for i in range(len_dw)])

    def initialize(self, sim):
        super().initialize(sim)
        if self.start is None:
            self.start = sim["start"]
        self.si = sc.findfirst(sim.res_yearvec, self.start)
        self.npts = len(sim.res_yearvec[self.si :])
        self.years = sim.res_yearvec[self.si :]
        self.yll = np.zeros(self.npts)
        self.yld = np.zeros(self.npts)
        self.dalys = 0
        return

    def apply(self, sim):

        if sim.yearvec[sim.t] >= self.start:
            ppl = sim.people
            li = np.floor(sim.yearvec[sim.t])
            idx = sc.findfirst(self.years, li)

            # Get new people with cancer and add up all their YLL and YLD now (incidence-based DALYs)
            new_cancers = ppl.date_cancerous == sim.t
            new_cancer_inds = hpv.true(new_cancers)
            if len(new_cancer_inds):
                self.yld[idx] += sum(
                    ppl.scale[new_cancer_inds]
                    * ppl.dur_cancer[new_cancers]
                    * self.av_disutility
                )
                age_death = ppl.age[new_cancer_inds] + ppl.dur_cancer[new_cancers]
                years_left = np.maximum(0, self.life_expectancy - age_death)
                self.yll[idx] += sum(ppl.scale[new_cancer_inds] * years_left)

        return

    def finalize(self, sim):
        self.dalys = np.sum(self.yll + self.yld)
        return



class econ_analyzer(hpv.Analyzer):
    """
    Analyzer for feeding into costing/health economic analysis.

    Produces a dataframe by year storing:

        - Resource use: number of vaccines, screens, lesions treated, cancers treated
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    """

    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        return

    def initialize(self, sim):
        super().initialize(sim)
        columns = [
            "new_hpv_screens",
            "new_vaccinations",
            "new_thermal_ablations",
            "new_leeps",
            "new_cancer_treatments",
        ]

        self.df = dict()
        for col in columns:
            self.df[col] = 0
        return

    def apply(self, sim):
        return

    def finalize(self, sim):
        for idx in range(sim.res_npts):

            # Pull out characteristics of sim to decide what resources we need
            simvals = sim.meta.vals
            scenario_label = simvals.scen
            
            # Check for routine vaccination intervention
            routine_intv = sim.get_intervention("Routine vx", die=False)
            if routine_intv is not None:
                self.df["new_vaccinations"] += routine_intv.n_products_used.values[idx]
            
            # Check for catchup vaccination intervention  
            catchup_intv = sim.get_intervention("Catchup vx", die=False)
            if catchup_intv is not None:
                self.df["new_vaccinations"] += catchup_intv.n_products_used.values[idx]

            # Check if this is a vaccination-only scenario (skip screening/treatment)
            is_vaccination_only = (scenario_label in ['90-0-0', '50-0-0'] or 
                                 'baseline' in scenario_label or 
                                 scenario_label.startswith('vx_') or
                                 scenario_label.startswith('test_') or
                                 'annual' in scenario_label or
                                 'biennial' in scenario_label or
                                 'triennial' in scenario_label or
                                 'quadrennial' in scenario_label or
                                 'quinquennial' in scenario_label)
            
            if not is_vaccination_only:
                self.df["new_hpv_screens"] += sim.get_intervention(
                            "screening"
                        ).n_products_used.values[idx]
                self.df["new_thermal_ablations"] += sim.get_intervention(
                        "ablation"
                    ).n_products_used.values[idx]
                self.df["new_leeps"] += sim.get_intervention(
                        "excision"
                    ).n_products_used.values[idx]
                self.df["new_cancer_treatments"] += sim.get_intervention(
                        "radiation"
                    ).n_products_used.values[idx]
            if 'HPV FASTER' in scenario_label:
                # add in HPV FASTER resources
                self.df["new_vaccinations"] += sim.get_intervention(
                    "HPV FASTER vx"
                ).n_products_used.values[idx]
 

        return


class genotype_analyzer(hpv.Analyzer):
    """
    Analyzer for tracking HPV incidence and cancer rates by genotype.
    
    Produces annual results for each genotype (16, 18, hi5, ohr):
    - New HPV infections by type
    - New cancer cases by type  
    - Cancer deaths by type
    - Prevalence by type
    """
    
    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.si = None  # Start index
        return
    
    def initialize(self, sim):
        super().initialize(sim)
        
        # Set start index based on start year
        if self.start is None:
            self.start = sim["start"]
        self.si = sc.findfirst(sim.res_yearvec, self.start)
        self.npts = len(sim.res_yearvec[self.si:])
        self.years = sim.res_yearvec[self.si:]
        
        # Get genotypes from simulation
        self.genotypes = sim['genotypes']
        
        # Initialize result arrays for each genotype
        self.results = {}
        for genotype in self.genotypes:
            geno_name = f"hpv{genotype}" if isinstance(genotype, int) else genotype
            self.results[geno_name] = {
                'new_infections': np.zeros(self.npts),
                'new_cancers': np.zeros(self.npts), 
                'cancer_deaths': np.zeros(self.npts),
                'prevalence': np.zeros(self.npts)
            }
        
        return
    
    def apply(self, sim):
        # Get current time point - check if we're in the tracking period
        if sim.yearvec[sim.t] < self.start:
            return
        
        # Find the index in our results arrays
        current_year = sim.yearvec[sim.t]
        ti = sc.findfirst(self.years, np.floor(current_year))
        if ti is None or ti >= self.npts:
            return
            
        people = sim.people
        
        # Process each genotype
        for gi, genotype in enumerate(self.genotypes):
            geno_name = f"hpv{genotype}" if isinstance(genotype, int) else genotype
            
            # Check if we have enough genotypes in the data
            if gi >= people.date_infectious.shape[1]:
                continue
                
            # New infections this timestep (people who became infected with this genotype)
            if hasattr(people, 'date_infectious'):
                newly_infected = people.date_infectious[:, gi] == sim.t
                self.results[geno_name]['new_infections'][ti] += newly_infected.sum()
            
            # Current prevalence (active infections of this genotype)
            if hasattr(people, 'infectious'):
                currently_infected = people.infectious[:, gi]
                self.results[geno_name]['prevalence'][ti] = currently_infected.sum()
            
            # New cancers this timestep (people who progressed to cancer with this genotype)
            if hasattr(people, 'date_cancerous'):
                newly_cancerous = people.date_cancerous[:, gi] == sim.t
                self.results[geno_name]['new_cancers'][ti] += newly_cancerous.sum()
            
            # Cancer deaths this timestep (attributed to this genotype)
            # Note: HPVsim doesn't track which genotype caused cancer death, so we estimate
            # based on the proportion of current cancerous cases by genotype
            if hasattr(people, 'date_dead_cancer') and hasattr(people, 'cancerous'):
                total_cancer_deaths = (people.date_dead_cancer == sim.t).sum()
                
                if total_cancer_deaths > 0:
                    # Calculate proportion of current cancerous cases by genotype
                    current_cancerous_by_genotype = people.cancerous[:, gi].sum()
                    total_current_cancerous = people.cancerous.sum()
                    
                    if total_current_cancerous > 0:
                        # Attribute deaths proportionally to current cancer burden by genotype
                        genotype_proportion = current_cancerous_by_genotype / total_current_cancerous
                        attributed_deaths = total_cancer_deaths * genotype_proportion
                    else:
                        # Fallback to equal distribution if no current cancerous cases
                        attributed_deaths = total_cancer_deaths / len(self.genotypes)
                        
                    self.results[geno_name]['cancer_deaths'][ti] += attributed_deaths
        
        return
    
    def finalize(self, sim):
        """Convert results to rates per 100,000 population"""
        
        # Get population size by time point for rate calculations
        pop_sizes = sim.results['n_alive'][self.si:]
        
        # Convert to rates per 100,000
        self.rates = {}
        for genotype in self.genotypes:
            geno_name = f"hpv{genotype}" if isinstance(genotype, int) else genotype
            self.rates[geno_name] = {}
            
            for metric in ['new_infections', 'new_cancers', 'cancer_deaths', 'prevalence']:
                # Calculate rate per 100,000 population
                self.rates[geno_name][f'{metric}_rate'] = (
                    self.results[geno_name][metric] / pop_sizes * 100000
                )
        
        # Store years for easy access
        self.rate_years = self.years
        
        return

