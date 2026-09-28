"""
Define the HPVsim simulation objects.
"""

# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import pars_data as dp
import utils as ut
import analyzers as an
import pars_scenarios as sp

# %% Settings and filepaths
# Debug switch
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


# %% Simulation creation functions
def make_sim(
    location=None,
    calib_pars=None,
    debug=0,
    screen_intv=None,
    vx_intv=None,
    analyzers=[],
    datafile=None,
    seed=1,
    econ_analyzer=False,
    end=2020,
    n_agents=10e3,
    ms_agent_ratio=100,
    dist_type="lognormal",
    marriage_scale=1,
    debut_bias=[0, 0],
):
    """Define parameters, analyzers, and interventions for the simulation -- not the sim itself"""

    pars = dict(
        n_agents=[n_agents, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=end,
        network="default",
        genotypes=[16, 18, "hi5", "ohr"],
        location=location,
        debut=ut.make_sb_data(
            location=location, dist_type=dist_type, debut_bias=debut_bias
        ),
        mixing=dp.mixing[location],
        layer_probs=dp.make_layer_probs(
            location=location, marriage_scale=marriage_scale
        ),
        f_partners=dp.f_partners,
        m_partners=dp.m_partners,
        init_hpv_dist=dp.init_genotype_dist[location],
        init_hpv_prev={
            "age_brackets": np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            "m": np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            "f": np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=ms_agent_ratio,
        verbose=0.0,
    )

    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    if analyzers is None:
        analyzers = sc.autolist()
    if econ_analyzer:
        analyzers += an.dalys(start=2020)
        analyzers += an.econ_analyzer()

    interventions = sc.autolist()
    if vx_intv is not None and len(vx_intv):
        interventions += sp.get_vx_intvs(**vx_intv)

    if screen_intv is not None and len(screen_intv):
        interventions += sp.get_screen_intvs(**screen_intv)


    sim = hpv.Sim(
        pars=pars,
        interventions=interventions,
        analyzers=analyzers,
        datafile=datafile,
        rand_seed=seed,
    )

    return sim


# %% Simulation running functions
def run_sim(
    location=None,
    screen_intv=None,
    vx_intv=None,
    analyzers=None,
    debug=0,
    seed=1,
    verbose=0.2,
    do_save=False,
    dist_type="lognormal",
    marriage_scale=1,
    debut_bias=[0, 0],
    econ_analyzer=False,
    end=2020,
    n_agents=10e3,
    ms_agent_ratio=100,
    calib_pars=None,
    meta=None,
):
    if analyzers is None:
        analyzers = sc.autolist()
    else:
        analyzers = sc.promotetolist(analyzers)

    dflocation = location.replace(" ", "_")

    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        analyzers=analyzers,
        screen_intv=screen_intv,
        vx_intv=vx_intv,
        econ_analyzer=econ_analyzer,
        dist_type=dist_type,
        marriage_scale=marriage_scale,
        debut_bias=debut_bias,
        calib_pars=calib_pars,
        end=end,
        n_agents=n_agents,
        ms_agent_ratio=ms_agent_ratio,
    )
    sim["rand_seed"] = seed
    sim.label = f"{location}--{seed}"

    # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta  # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location  # Store location in an easy-to-access place

    # Run
    sim["verbose"] = verbose
    sim.run()
    sim.shrink()

    if do_save:
        sim.save(f"results/{dflocation}.sim")

    return sim


def run_sims(
    locations=None,
    age_pyr=True,
    debug=False,
    verbose=-1,
    analyzers=None,
    dist_type="lognormal",
    marriage_scale=1,
    debut_bias=[0, 0],
    calib_par_stem=None,
    do_save=False,
    *args,
    **kwargs,
):
    """Run multiple simulations in parallel"""

    kwargs = sc.mergedicts(
        dict(
            debug=debug,
            verbose=verbose,
            analyzers=analyzers,
            dist_type=dist_type,
            age_pyr=age_pyr,
            marriage_scale=marriage_scale,
            calib_par_stem=calib_par_stem,
            debut_bias=debut_bias,
        ),
        kwargs,
    )
    simlist = sc.parallelize(
        run_sim,
        iterkwargs=dict(location=locations),
        kwargs=kwargs,
        serial=debug,
        die=True,
    )
    sims = sc.objdict(
        {location: sim for location, sim in zip(locations, simlist)}
    )  # Convert from a list to a dict

    if do_save:
        for loc, sim in sims.items():
            sim.save(f"results/{loc}.sim")

    return sims


def run_parsets(
    location=None, debug=False, verbose=0.1, analyzers=None, save_results=True, **kwargs
):
    """Run multiple simulations in parallel"""

    dflocation = location.replace(" ", "_")
    parsets = sc.loadobj(f"results/immunovarying/{dflocation}_pars_jun15_iv_all.obj")
    kwargs = sc.mergedicts(
        dict(location=location, debug=debug, verbose=verbose, analyzers=analyzers),
        kwargs,
    )
    simlist = sc.parallelize(
        run_sim,
        iterkwargs=dict(calib_pars=parsets),
        kwargs=kwargs,
        serial=debug,
        die=True,
    )
    msim = hpv.MultiSim(simlist)
    msim.reduce()
    if save_results:
        sc.saveobj(f"results/msims/{dflocation}.obj", msim.results)

    return msim


# %% Run as a script
if __name__ == "__main__":
    T = sc.timer()

    location = "india"  # , 'ethiopia', 'drc']  #loc.locations
    calib_par_stem = "_nov13"
    calib_pars = sc.loadobj(f"results/{location}_pars{calib_par_stem}.obj")
    sim = run_sim(
        location=location,
        calib_pars=calib_pars,
        end=2032,
    )
    sim.plot()

    T.toc("Done")
