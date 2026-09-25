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
import pylab as pl
import seaborn as sns
import numpy as np
import pandas as pd

# Imports from this repository
import run_sim as rs
import utils as ut

# CONFIGURATIONS TO BE SET BY USERS BEFORE RUNNING
to_run = [
    #'run_calibration',  # Make sure this is uncommented if you want to _run_ the calibrations (usually on VMs)
    "plot_calibration",  # Make sure this is uncommented if you want to _plot_ the calibrations (usually locally)
]
debug = (
    True  # If True, this will do smaller runs that can be run locally for debugging
)
do_save = True

# Run settings for calibration (dependent on debug)
n_trials = [8000, 10000][debug]  # How many trials to run for calibration
n_workers = [8, 8][debug]  # How many cores to use
storage = ["mysql://hpvsim_user@localhost/hpvsim_db", None][
    debug
]  # Storage for calibrations


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


########################################################################
# Load pre-run calibration
########################################################################
def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True, filestem=""):
    fnlocation = location.replace(" ", "_").replace("'", "_")
    filename = f"{fnlocation}_calib{filestem}"
    calib = sc.load(f"results/{filename}.obj")
    if do_plot:
        sc.fonts(add=sc.thisdir(aspath=True) / "Libertinus Sans")
        sc.options(font="Libertinus Sans")
        fig = calib.plot(res_to_plot=200, plot_type="sns.boxplot", do_save=False)
        fig.suptitle(f"Calibration results, {location.capitalize()}")
        fig.tight_layout()
        fig.savefig(f"figures/{filename}.png")

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        trial_pars = sc.autolist()
        for i in range(100):
            trial_pars += calib.trial_pars_to_sim_pars(which_pars=i)
        sc.save(f"results/{location}_pars{filestem}.obj", calib_pars)

    return calib


def plot_calibration_combined(calibs, locations, res_to_plot=50):
    ut.set_font(12)
    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    axes = axes.flatten()
    resname = "cancers"
    date = 2020

    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            location = locations[pn]

            calib = calibs[pn]

            # Pull out model results and data
            reslist = calib.analyzer_results
            target_data = calib.target_data[0]
            target_data = target_data[(target_data.name == resname)]

            # Make labels
            baseres = reslist[0]["cancers"]
            age_labels = [
                str(int(baseres["bins"][i])) + "-" + str(int(baseres["bins"][i + 1]))
                for i in range(len(baseres["bins"]) - 1)
            ]
            age_labels.append(str(int(baseres["bins"][-1])) + "+")

            # Pull out results to plot
            plot_indices = calib.df.iloc[0:res_to_plot, 0].values
            res = [reslist[i] for i in plot_indices]

            # Plot data
            x = np.arange(len(age_labels))
            ydata = np.array(target_data.value)
            ax.scatter(x, ydata, color="k", marker="s", label="Data")

            # Construct a dataframe with things in the most logical order for plotting
            bins = []
            values = []
            for run_num, run in enumerate(res):
                bins += x.tolist()
                values += list(run[resname][date])
            modeldf = pd.DataFrame({"bins": bins, "values": values})
            sns.boxplot(
                ax=ax,
                x="bins",
                y="values",
                data=modeldf,
                color="b",
                boxprops=dict(alpha=0.4),
            )

            # Set title and labels
            # ax.set_xlabel('Age group')
            title_country = location.title()
            if title_country == "Drc":
                title_country = "DRC"
            if title_country == "Cote Divoire":
                title_country = "Cote d'Ivoire"
            ax.set_title(title_country)
            ax.set_ylabel("")
            ax.set_xlabel("")
            # ax.legend()
            if pn in [6, 7, 8]:
                stride = np.arange(0, len(baseres["bins"]), 2)
                ax.set_xticks(x[stride], baseres["bins"].astype(int)[stride])
            else:
                ax.set_xticks(x, [])

    fig.suptitle("Cervical cancer cases in 2020")
    fig.tight_layout()
    pl.savefig(f"figures/calibration_combined.png", dpi=100)


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
        filestem = "_jul22"
        for location in locations:
            sim, calib = run_calib(
                location=location,
                n_trials=n_trials,
                n_workers=n_workers,
                do_save=do_save,
                do_plot=False,
                filestem=filestem,
            )

    # Load the calibration, plot it, and save the best parameters -- usually locally
    if "plot_calibration" in to_run:
        calibs = []
        for location in locations:
            if location in ["tanzania", "myanmar"]:
                filestem = "_nov07"
            elif location in ["uganda", "drc"]:
                filestem = "_nov06"
            else:
                filestem = "_jul22"
            calib = load_calib(
                location=location, do_plot=True, save_pars=False, filestem=filestem
            )
            calibs.append(calib)
        plot_calibration_combined(calibs, locations)

    T.toc("Done")
