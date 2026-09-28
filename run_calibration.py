"""
This file is used to run calibrations for TxV 10-country analysis.

Instructions: Go to the CONFIGURATIONS section on lines 29-36 to set up the script before running it.
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
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut

# CONFIGURATIONS TO BE SET BY USERS BEFORE RUNNING
to_run = [
    'run_calibration',
]
debug = (
    True  # If True, this will do smaller runs that can be run locally for debugging
)
do_save = True

# Run settings for calibration (dependent on debug)
if debug:
    n_trials = 8000
    n_workers = 8
    storage = None
else:
    n_trials = 10000
    n_workers = 8
    storage = "mysql://hpvsim_user@localhost/hpvsim_db"


########################################################################
# Run calibration
########################################################################
def make_priors():
    default = dict(
        rel_beta=[0.9, 0.7, 1.3, 0.05],
        cancer_fn=dict(ld50=[20, 10, 35, 0.5]),
        dur_cin=dict(par1=[7, 2, 15, 0.1], par2=[15, 5, 30, 0.5]),
    )

    genotype_pars = dict(
        hpv18=sc.dcp(default),
        hi5=sc.dcp(default),
        ohr=sc.dcp(default),
        hpv16=dict(
            cancer_fn=dict(ld50=[20, 10, 35, 0.5]),
            dur_cin=dict(par1=[7, 2, 14, 0.1], par2=[15, 5, 30, 0.5]),
        ),
    )

    return genotype_pars


def run_calib(
    location=None,
    n_trials=None,
    n_workers=None,
    do_plot=False,
    do_save=True,
    filestem="",
):

    sim = rs.make_sim(location)
    datafiles = ut.make_datafiles([location])[location]

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.06, 0.01, 0.49, 0.02],
        own_imm_hr=[0.5, 0.1, 1, 0.05],
        age_risk=dict(risk=[1, 0.5, 4, 0.1], age=[30, 20, 50, 1]),
        sev_dist=dict(par1=[1, 0.5, 2, 0.1]),
        cell_imm_init=dict(par1=[0.5, 0.1, 0.9, 0.05]),
    )

    if location == "nigeria":
        calib_pars["sev_dist"]["par1"] = [2, 1, 3, 0.1]
    # if location == 'india':
    if location is None:
        sexual_behavior_pars = dict(
            m_cross_layer=[0.9, 0.5, 0.95, 0.05],
            m_partners=dict(c=dict(par1=[10, 5, 12, 1])),
            f_cross_layer=[0.1, 0.05, 0.5, 0.05],
            f_partners=dict(c=dict(par1=[1, 0.5, 2, 0.1], par2=[0.2, 0.1, 1, 0.05])),
        )
    else:
        sexual_behavior_pars = dict(
            m_cross_layer=[0.3, 0.05, 0.9, 0.05],
            m_partners=dict(c=dict(par1=[0.2, 0.05, 0.89, 0.02])),
            f_cross_layer=[0.1, 0.05, 0.9, 0.05],
            f_partners=dict(c=dict(par1=[0.2, 0.05, 0.89, 0.02])),
        )
    calib_pars = sc.mergedicts(calib_pars, sexual_behavior_pars)

    genotype_pars = make_priors()

    # Save some extra sim results
    extra_sim_result_keys = ["cancers", "cancer_incidence", "asr_cancer_incidence"]

    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars,
        genotype_pars=genotype_pars,
        name=f"{location}_calib_final",
        datafiles=datafiles,
        extra_sim_result_keys=extra_sim_result_keys,
        total_trials=n_trials,
        n_workers=n_workers,
        storage=storage,
    )
    calib.calibrate()
    filename = f"{location}_calib{filestem}"
    if do_plot:
        calib.plot(do_save=True, fig_path=f"figures/{filename}.png")
    if do_save:
        sc.saveobj(f"results/{filename}.obj", calib)

    print(f"Best pars are {calib.best_pars}")

    return sim, calib




# %% Run as a script
if __name__ == "__main__":

    T = sc.timer()
    locations = [
        'cote d\'ivoire',
        'zambia',
        'sierra leone',
    ]

    # Run calibration - usually on VMs
    if "run_calibration" in to_run:
        filestem = "_nov06"
        for location in locations:
            sim, calib = run_calib(
                location=location,
                n_trials=n_trials,
                n_workers=n_workers,
                do_save=do_save,
                do_plot=False,
                filestem=filestem,
            )

    T.toc("Done")
