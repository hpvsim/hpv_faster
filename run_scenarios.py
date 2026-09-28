"""
Run HPVsim scenarios for each location. 

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
"""

# %% General settings

import os

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

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp
import analyzers as an

# Comment out to not run
to_run = [
    "run_scenarios",
    # 'plot_scenarios'
]

# Comment out locations to not run
locations = [
    "india",  # 0
    "indonesia",  # 1
    "nigeria",  # 2
    "tanzania",  # 3
    "bangladesh",  # 4
    "myanmar",  # 5
    "uganda",  # 6
    "ethiopia",  # 7
    "drc",  # 8
    # 'kenya'         # 9
]

debug = 0
n_seeds = [3, 1][debug]  # How many seeds to run per cluster

# %% Functions


def make_msims(sims, use_mean=True):
    """
    Utility to take a slice of sims and turn it into a multisim
    """

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_sc, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except parameter set matches
        assert i_sc == sim.meta.inds[0]
        assert (s == 0) or i_s != sim.meta.inds[1]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_sc]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop("seed")

    print(f"Processing multisim {msim.meta.vals.values()}...")

    return msim


def run_scens(
    location=None,
    scenarios=None,  # Input data
    debug=0,
    n_seeds=n_seeds,
    verbose=-1,  # Sim settings
    calib_filestem="",
    filestem="",  # Output settings
):
    """
    Run all screening/triage product scenarios for a given location
    """

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(scenarios) * n_seeds

    for i_sc, sc_label, scenario_pars in scenarios.enumitems():
        for i_s in range(n_seeds):  # n samples per cluster
            count += 1
            meta = sc.objdict()
            meta.count = count
            meta.n_sims = n_sims
            meta.inds = [i_sc, i_s]
            meta.vals = sc.objdict(
                    sc.mergedicts(
                            scenario_pars['screen_scen'],
                            scenario_pars['vx_scen'],
                            dict(
                                seed=i_s,
                                scen=sc_label,
                            ),
                        )
                    )
            ikw.append(
                        sc.objdict(
                            screen_intv=scenario_pars['screen_scen'],
                            vx_intv=scenario_pars['vx_scen'],
                            seed=i_s,
                        )
                    )
            ikw[-1].meta = meta

    # Actually run
    sc.heading(f"Running {len(ikw)} scenario sims...")
    calib_pars = sc.loadobj(f"results/{location}_pars{calib_filestem}.obj")
    end = 2060
    kwargs = dict(
        calib_pars=calib_pars,
        verbose=verbose,
        debug=debug,
        location=location,
        econ_analyzer=True,
        end=end,
        n_agents=50e3,
    )
    n_workers = 40
    all_sims = sc.parallelize(
        rs.run_sim, iterkwargs=ikw, kwargs=kwargs, ncpus=n_workers
    )

    products = [
        "new_hpv_screens",
        "new_vaccinations",
        "new_thermal_ablations",
        "new_leeps",
        "new_cancer_treatments",
    ]

    # Rearrange sims
    sims = np.empty(
        (len(scenarios), n_seeds), dtype=object
    )
    econdfs = sc.autolist()
    for sim in all_sims:  # Unflatten array
        i_sc, i_s = sim.meta.inds
        sims[i_sc, i_s] = sim
        econdf = pd.DataFrame()
        if i_s == 0:
            product_res = sim.get_analyzer(an.econ_analyzer).df
            for prod in products:
                econdf[prod] = [product_res[prod]]
            econdf["dalys"] = sim.get_analyzer(an.dalys).dalys
            econdf["location"] = location
            econdf["seed"] = i_s
            econdf["scenario"] = sim.meta.vals["scen"]
            econdfs += econdf
        sim["analyzers"] = []  # Remove the analyzer so we don't need to reduce it
    econ_df = pd.concat(econdfs)
    sc.saveobj(f"{ut.resfolder}/{location}{filestem}_econ.obj", econ_df)

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_sc in range(len(scenarios)):
        sim_seeds = sims[i_sc, :].tolist()
        all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(scenarios)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)
    print("finished making msims")

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_sc = msim.meta.inds
        msims[i_sc] = msim
        df = pd.DataFrame()
        df["year"] = msim.results["year"]
        df["infections"] = msim.results["infections"][:]
        df["infections_low"] = msim.results["infections"].low
        df["infections_high"] = msim.results["infections"].high
        df["cancers"] = msim.results["cancers"][:]
        df["cancers_low"] = msim.results["cancers"].low
        df["cancers_high"] = msim.results["cancers"].high
        df["cancer_incidence"] = msim.results["cancer_incidence"][:]
        df["cancer_incidence_high"] = msim.results["cancer_incidence"].high
        df["cancer_incidence_low"] = msim.results["cancer_incidence"].low
        df["asr_cancer_incidence"] = msim.results["asr_cancer_incidence"][:]
        df["asr_cancer_incidence_low"] = msim.results["asr_cancer_incidence"].low
        df["asr_cancer_incidence_high"] = msim.results["asr_cancer_incidence"].high
        df["cancer_deaths"] = msim.results["cancer_deaths"][:]
        df["cancer_deaths_low"] = msim.results["cancer_deaths"].low
        df["cancer_deaths_high"] = msim.results["cancer_deaths"].high
        df["n_screened"] = msim.results["n_screened"][:]
        df["n_screened_low"] = msim.results["n_screened"].low
        df["n_screened_high"] = msim.results["n_screened"].high
        df["n_cin_treated"] = msim.results["n_cin_treated"][:]
        df["n_cin_treated_low"] = msim.results["n_cin_treated"].low
        df["n_cin_treated_high"] = msim.results["n_cin_treated"].high
        df["n_vaccinated"] = msim.results["n_vaccinated"][:]
        df["n_vaccinated_low"] = msim.results["n_vaccinated"].low
        df["n_vaccinated_high"] = msim.results["n_vaccinated"].high
        df["new_screens"] = msim.results["new_screens"][:]
        df["new_cin_treatments"] = msim.results["new_cin_treatments"][:]
        df["new_cancer_treatments"] = msim.results["new_cancer_treatments"][:]
        df["location"] = location

        # Store metadata about run
        df["scenario"] = msim.meta.vals["scen"]
        dfs += df

    alldf = pd.concat(dfs)
    sc.saveobj(f"{ut.resfolder}/{location}{filestem}.obj", alldf)

    return alldf, msims


# %% Run as a script
if __name__ == "__main__":

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)

    if "run_scenarios" in to_run:
        for location in locations:
            if location in ["tanzania", "myanmar"]:
                calib_filestem = "_nov07"
            elif location in ["uganda", "drc"]:
                calib_filestem = "_nov06"
            else:
                calib_filestem = "_nov13"

            # Construct the scenarios
            # 1. 90% vx coverage of 9-14 year olds
            # 2. 90% vx coverage of 9-14 year olds + screening with 70% coverage
            # 3. 90% vx coverage of 9-14 year olds + HPV FASTER: 90% coverage of 22-40 year olds after S&T 

            hpv_screen = dict(precin=0.45, cin=0.95, cancerous=0.95)
            vx_scen = {
                "90% coverage": dict(
                        vx_coverage=0.9,
                        age_range=(9, 14),
                        start_year=2026,
                    ),
                "50% coverage": dict(
                        vx_coverage=0.5,
                        age_range=(9, 14),
                        start_year=2026,
                    ),
            }
            screen_scens = sc.objdict(
                {
                    "No screening": {},
                    "70% coverage, 10% LTFU": dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.1,
                    ),
                    "30% coverage, 10% LTFU": dict(
                        primary=hpv_screen,
                        screen_coverage=0.3,
                        start_year=2026,
                        ltfu=0.1,
                    ),
                    "50% coverage, 10% LTFU": dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.1,
                    ),
                    'HPV FASTER, 22-50, 70% coverage, 10% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.1,
                        age_range=(22, 50),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-50, 70% coverage, 1% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.01,
                        age_range=(22, 50),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-50, 70% coverage, 30% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.3,
                        age_range=(22, 50),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-50, 50% coverage, 10% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.1,
                        age_range=(22, 50),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-50, 50% coverage, 1% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.01,
                        age_range=(22, 50),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-50, 50% coverage, 30% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.3,
                        age_range=(22, 50),
                        paired_px=True,
                    ),
                    
                    'HPV FASTER, 22-30, 70% coverage, 10% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.1,
                        age_range=(22, 30),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-30, 70% coverage, 1% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.01,
                        age_range=(22, 30),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-30, 70% coverage, 30% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.3,
                        age_range=(22, 30),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-30, 50% coverage, 10% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.1,
                        age_range=(22, 30),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-30, 50% coverage, 1% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.01,
                        age_range=(22, 30),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-30, 50% coverage, 30% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.3,
                        age_range=(22, 30),
                        paired_px=True,
                    ),
                    
                    'HPV FASTER, 22-40, 70% coverage, 10% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.1,
                        age_range=(22, 40),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-40, 70% coverage, 1% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.01,
                        age_range=(22, 40),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-40, 70% coverage, 30% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.7,
                        start_year=2026,
                        ltfu=0.3,
                        age_range=(22, 40),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-40, 50% coverage, 10% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.1,
                        age_range=(22, 40),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-40, 50% coverage, 1% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.01,
                        age_range=(22, 40),
                        paired_px=True,
                    ),
                    'HPV FASTER, 22-40, 50% coverage, 30% LTFU': dict(
                        primary=hpv_screen,
                        screen_coverage=0.5,
                        start_year=2026,
                        ltfu=0.3,
                        age_range=(22, 40),
                        paired_px=True,
                    ),
                    
                }
            )
            scenarios = sc.objdict(
                {
                    "90-0-0": sc.objdict(
                        screen_scen=screen_scens["No screening"],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "90-70-90": sc.objdict(
                        screen_scen=screen_scens["70% coverage, 10% LTFU"],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "90-50-90": sc.objdict(
                        screen_scen=screen_scens["50% coverage, 10% LTFU"],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "90-30-90": sc.objdict(
                        screen_scen=screen_scens["30% coverage, 10% LTFU"],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-50, 70% coverage, 10% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-50, 70% coverage, 10% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-50, 70% coverage, 30% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-50, 70% coverage, 30% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-50, 50% coverage, 10% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-50, 50% coverage, 10% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-50, 50% coverage, 30% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-50, 50% coverage, 30% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-40, 70% coverage, 10% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-40, 70% coverage, 10% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-40, 70% coverage, 30% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-40, 70% coverage, 30% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-40, 50% coverage, 10% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-40, 50% coverage, 10% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-40, 50% coverage, 30% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-40, 50% coverage, 30% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-30, 70% coverage, 10% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-30, 70% coverage, 10% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-30, 70% coverage, 30% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-30, 70% coverage, 30% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-30, 50% coverage, 10% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-30, 50% coverage, 10% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                    "HPV FASTER, 22-30, 50% coverage, 30% LTFU": sc.objdict(
                        screen_scen=screen_scens['HPV FASTER, 22-30, 50% coverage, 30% LTFU'],
                        vx_scen=vx_scen["90% coverage"],
                    ),
                }
            )
            
            alldf, msims = run_scens(
                scenarios=scenarios,
                location=location,
                debug=debug,
                calib_filestem=calib_filestem,
                filestem="_june25",
            )

