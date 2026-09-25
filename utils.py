'''
Utilities for multicalibration
'''

# Standard imports
import sciris as sc
import hpvsim as hpv
import hpvsim.parameters as hppar
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec
import math
from scipy.stats import lognorm, norm

import analyzers as an
import run_sim as rs

resfolder = 'results'
figfolder = 'figures'
datafolder = 'data'

def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


def map_sb_loc(location):
    ''' Map between different representations of country names '''
    location = location.title()
    if location == "Cote Divoire": location = "Cote d'Ivoire"
    if location == "Cote D'Ivoire": location = "Cote d'Ivoire"  # Fix capitalization issue
    if location == "Cote D'Ivoire Denguele": location = "Cote d'Ivoire Denguele"  # Fix capitalization issue
    if location == "Cote D'Ivoire Ultra High Risk": location = "Cote d'Ivoire Ultra High Risk"  # Fix capitalization issue
    if location == "Drc": location = 'Congo Democratic Republic'
    return location


def rev_map_sb_loc(location):
    ''' Map between different representations of country names '''
    location = location.lower()
    # location = location.replace(' ', '_')
    if location == 'congo democratic republic': location = "drc"
    if location == "cote d'ivoire": location = 'cote divoire'
    return location


def make_sb_data(location=None, dist_type='lognormal', debut_bias=[0,0]):

    sb_location = map_sb_loc(location)

    # Read in data
    sb_data_f = pd.read_csv(f'data/sb_pars_women_{dist_type}.csv')
    sb_data_m = pd.read_csv(f'data/sb_pars_men_{dist_type}.csv')

    try:
        distf = sb_data_f.loc[sb_data_f["location"]==sb_location,"dist"].iloc[0]
        par1f = sb_data_f.loc[sb_data_f["location"]==sb_location,"par1"].iloc[0]
        par2f = sb_data_f.loc[sb_data_f["location"]==sb_location,"par2"].iloc[0]
        distm = sb_data_m.loc[sb_data_m["location"]==sb_location,"dist"].iloc[0]
        par1m = sb_data_m.loc[sb_data_m["location"]==sb_location,"par1"].iloc[0]
        par2m = sb_data_m.loc[sb_data_m["location"]==sb_location,"par2"].iloc[0]
    except:
        print(f'No data for {sb_location=}, {location=}')

    debut = dict(
        f=dict(dist=distf, par1=par1f+debut_bias[0], par2=par2f),
        m=dict(dist=distm, par1=par1m+debut_bias[1], par2=par2m),
    )

    return debut


def make_datafiles(locations):
    ''' Get the relevant datafiles for the selected locations '''
    datafiles = dict()
    asr_locs            = ['drc', 'ethiopia', 'kenya', 'nigeria', 'tanzania', 'uganda', 'zambia', 'cote d\'ivoire', 'sierra leone']
    cancer_type_locs    = ['ethiopia', 'kenya', 'nigeria', 'tanzania', 'india', 'uganda']
    cin_type_locs       = ['nigeria', 'tanzania', 'india', 'cote d\'ivoire', 'zambia']

    for location in locations:
        dflocation = location.replace(' ','_')
        datafiles[location] = [
            f'data/{dflocation}_cancer_cases.csv',
        ]

        if location in asr_locs:
            datafiles[location] += [f'data/{dflocation}_asr_cancer_incidence.csv']

        if location in cancer_type_locs:
            datafiles[location] += [f'data/{dflocation}_cancer_types.csv']

        if location in cin_type_locs:
            datafiles[location] += [f'data/{dflocation}_cin_types.csv']

    return datafiles



def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale


def plot_residual_burden(location=None, scens=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=24)

    r1 = np.arange(len(scens))
    bigdf = sc.loadobj(f'{resfolder}/{location}.obj')

    colors = sc.gridcolors(20)

    fig, axes = pl.subplots(2, 1, figsize=(20, 16))
    ib_labels = scens.keys()


    for ib, (ib_label, ib_scens) in enumerate(scens.items()):
        vx_scen_label = ib_scens[0]
        screen_scen_label = ib_scens[1]
        txvx_scen_label = ib_scens[2]
        df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                   & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
            ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high', 'cancers', 'cancers_low',
             'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high']].sum()
        ys = sc.findinds(df.index, 2020)[0]
        ye = sc.findinds(df.index, 2060)[0]
        years = np.array(df.index)[ys:ye]
        best_ts = np.array(df['cancers'])[ys:ye]
        low_ts = np.array(df['cancers_low'])[ys:ye]
        high_ts = np.array(df['cancers_high'])[ys:ye]


        axes[0].plot(years, best_ts, color=colors[ib], linewidth=3, label=ib_label)
        axes[0].fill_between(years, low_ts, high_ts, color=colors[ib], alpha=0.3)
        best_short = np.sum(np.array(df['cancers'])[ys:ye])
        low_short = np.sum(np.array(df['cancers_low'])[ys:ye])
        high_short = np.sum(np.array(df['cancers_high'])[ys:ye])
        yerr_short = abs(high_short - low_short)
        axes[1].barh(r1[ib], best_short, xerr=yerr_short, color=colors[ib])

    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    axes[1].set_yticks([r for r in range(len(r1))], ib_labels)
    axes[0].legend(loc=3)
    # axes[1, 0].legend()
    axes[0].set_ylabel('Cervical cancer cases')
    # axes[1].set_xlabel('Residual cervical cancer cases (2025-2060)')
    axes[1].set_xlabel('Cervical cancer deaths (2025-2060)')
    axes[0].set_ylim(bottom=0)
    axes[1].invert_yaxis()
    fig.suptitle(f'{location.capitalize()}')
    fig.tight_layout()
    fig_name = f'{figfolder}/{location}_residual_burden.png'
    sc.savefig(fig_name, dpi=100)
    return

def plot_VIMC_compare(locations=None, scens=None):

    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{resfolder}/{location}.obj')
        dfs += df

    bigdf = pd.concat(dfs)
    VIMC = pd.read_csv(f'{datafolder}/VIMC.csv')

    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    set_font(12)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    axes = axes.flatten()

    VIMC_pivot = pd.pivot_table(
        VIMC,
        values='cases',
        index='year',
        columns='country_name',
        aggfunc='sum'
    )
    VIMC_pivot= VIMC_pivot[20:61]
    vx_scen_label = scens[0]
    screen_scen_label = scens[1]
    txvx_scen_label = scens[2]
    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            location = locations[pn]

            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                       & (bigdf.txvx_scen == txvx_scen_label) & (bigdf.location == location)].groupby('year')[
                ['cancers', 'cancers_high', 'cancers_low']].sum()[2020:]

            years = np.array(df.index)
            ax.plot(years, df['cancers'], label='HPVsim')
            ax.fill_between(years, df['cancers_low'], df['cancers_high'], alpha=0.3)
            title_country = location.title()
            if title_country == 'Tanzania':
                title_country = 'Tanzania, United Republic of'

            if title_country == 'Drc':
                title_country = "Congo, the Democratic Republic of the"

            ax.plot(years, VIMC_pivot[title_country], label='VIMC')
            ax.set_ylim(bottom=0)
            ax.legend()
            sc.SIticks(ax)
            ax.set_title(location.capitalize())
    fig.suptitle('Cervical cancer cases over time')
    fig.tight_layout()
    fig_name = f'{figfolder}/VIMC_compare.png'
    sc.savefig(fig_name, dpi=100)

    return


def plot_txv_impact(location=None, background_scens=None, txvx_ages=None, txvx_efficacies=None, discounting=False):

    set_font(size=24)

    bigdf = sc.loadobj(f'{resfolder}/{location}.obj')
    econdf = sc.loadobj(f'{resfolder}/{location}_econ.obj')

    colors = sc.gridcolors(20)
    width = 0.1
    standard_le = 88.8

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']

    fig, axes = pl.subplots(2, 2, figsize=(16, 16))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                         & (bigdf.txvx_scen == 'No TxV')].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',
             'n_tx_vaccinated']].sum()

        NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                      & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['new_cancers', 'new_cancer_deaths']].sum()

        NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                              & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()

        if discounting:
            cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
            cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
        else:
            cancers = NoTxV_econdf_cancers['new_cancers'].values
            cancer_deaths = NoTxV_econdf_cancers['new_cancer_deaths'].values

        avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
        yll = np.sum((standard_le-avg_age_ca_death) * cancer_deaths)
        daly_noTxV = yll + yld

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df['cancers'])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df['cancers_low'])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df['cancers_high'])[ys:ye])

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df['cancer_deaths'])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(np.array(NoTxV_df['cancer_deaths_low'])[ys:ye])
        NoTxV_cancer_deaths_short_high = np.sum(np.array(NoTxV_df['cancer_deaths_high'])[ys:ye])

        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            for i_age, txvx_age in enumerate(txvx_ages):
                txvx_scen_label_age = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'

                TxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label_age)].groupby(
                    'year')[
                    ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                     'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high', 'n_tx_vaccinated']].sum()

                TxV_econdf_cancers = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths']].sum()

                TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                    & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                if discounting:
                    cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                    cancer_deaths = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
                else:
                    cancers = TxV_econdf_cancers['new_cancers'].values
                    cancer_deaths = TxV_econdf_cancers['new_cancer_deaths'].values

                avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                daly_TxV = yll + yld
                dalys_averted = daly_noTxV - daly_TxV
                best_TxV_cancers_short = np.sum(np.array(TxV_df['cancers'])[ys:ye])
                averted_cancers_short = NoTxV_cancers - best_TxV_cancers_short
                perc_cancers_averted_short = 100 * averted_cancers_short / NoTxV_cancers
                to_plot_short = perc_cancers_averted_short

                TxV_cancers = np.sum(np.array(TxV_df['cancers'])[ys:ye])
                TxV_cancers_low = np.sum(np.array(TxV_df['cancers_low'])[ys:ye])
                TxV_cancers_high = np.sum(np.array(TxV_df['cancers_high'])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(np.array(TxV_df['cancer_deaths'])[ys:ye])
                best_TxV_cancer_deaths_short_high = np.sum(np.array(TxV_df['cancer_deaths_high'])[ys:ye])
                best_TxV_cancer_deaths_short_low = np.sum(np.array(TxV_df['cancer_deaths_low'])[ys:ye])

                averted_cancer_deaths_short = NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                averted_cancer_deaths_short_high = NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                averted_cancer_deaths_short_low = NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low

                TxV_cancers_averted = NoTxV_cancers - TxV_cancers

                if i_eff + ib == 0:
                    axes[0,0].scatter(xes[i_eff][ib], TxV_cancers_averted, marker=markers[i_age], color=colors[i_eff],
                                    s=300, label=f'Age {txvx_age}')
                else:
                    axes[0,0].scatter(xes[i_eff][ib], TxV_cancers_averted, marker=markers[i_age], color=colors[i_eff],
                                    s=300)
                if ib + i_age == 0:
                    axes[0,1].scatter(xes[i_eff][ib], to_plot_short, marker=markers[i_age], color=colors[i_eff], s=300,
                                       label=txvx_efficacy)
                else:
                    axes[0,1].scatter(xes[i_eff][ib], to_plot_short, marker=markers[i_age], color=colors[i_eff], s=300)
                axes[1,0].scatter(xes[i_eff][ib], averted_cancer_deaths_short, marker=markers[i_age], color=colors[i_eff], s=300)
                axes[1,1].scatter(xes[i_eff][ib], dalys_averted, marker=markers[i_age], color=colors[i_eff], s=300)

        ib_labels = background_scens.keys()

        axes[1,0].set_xticks([r + 1.5*width for r in range(len(r1))], ib_labels)
        axes[1,1].set_xticks([r + 1.5 * width for r in range(len(r1))], ib_labels)
        axes[0,0].get_xaxis().set_visible(False)
        axes[0,1].get_xaxis().set_visible(False)
    axes[0,0].set_ylim(bottom=0)
    axes[0, 1].set_ylim(bottom=0)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_ylim(bottom=0)
    sc.SIticks(axes[0,0])
    sc.SIticks(axes[0,1])
    sc.SIticks(axes[1,0])
    sc.SIticks(axes[1,1])

    axes[0,0].set_ylabel(f'Cervical cancer cases averted (2030-2060)')
    axes[0,1].set_ylabel(f'Percent cervical cancer cases averted (2030-2060)')
    axes[1,0].set_ylabel(f'Deaths averted')
    axes[1,1].set_ylabel(f'DALYs averted')

    # axes[1].set_ylim([0, 30])
    axes[0,0].legend(title='Age of TxV')
    axes[0,1].legend(title='TxV Effectiveness')
    fig.suptitle(f'{location.capitalize()}')

    # axes[1].set_xlabel('Background intervention scenario')
    axes[1,0].set_xlabel('Background intervention scenario')
    axes[1, 1].set_xlabel('Background intervention scenario')
    fig.tight_layout()
    fig_name = f'{figfolder}/{location}_txv_impact.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_residual_burden_combined(locations, scens):

    set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{resfolder}/{location}.obj')
        dfs += df
    bigdf = pd.concat(dfs)
    colors = ['gold', 'orange', 'red', 'darkred']

    fig = pl.figure(constrained_layout=True, figsize=(16, 20))
    spec2 = GridSpec(ncols=3, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :])
    axes = []
    for i in np.arange(1,4):
        for j in np.arange(0,3):
            ax = fig.add_subplot(spec2[i,j])
            axes.append(ax)
    txvx_scen_label = 'No TxV'
    for ib, (ib_label, ib_scens) in enumerate(scens.items()):
        vx_scen_label = ib_scens[0]
        screen_scen_label = ib_scens[1]
        df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                   & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                 ['cancers', 'cancers_low', 'cancers_high']].sum()[2020:]
        years = np.array(df.index)
        ax1.plot(years, df['cancers'], color=colors[ib], label=ib_label)
        ax1.fill_between(years, df['cancers_low'], df['cancers_high'], color=colors[ib], alpha=0.3)

    for pn, location in enumerate(locations):
        ax = axes[pn]
        for ib, (ib_label, ib_scens) in enumerate(scens.items()):
            vx_scen_label = ib_scens[0]
            screen_scen_label = ib_scens[1]
            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                       & (bigdf.txvx_scen == txvx_scen_label) & (bigdf.location == location)].groupby('year')[
                     ['cancers', 'cancers_low', 'cancers_high']].sum()[2020:]
            years = np.array(df.index)
            ax.plot(years, df['cancers'], color=colors[ib])
            ax.fill_between(years, df['cancers_low'], df['cancers_high'], color=colors[ib], alpha=0.3)
            sc.SIticks(ax)
            if location == 'drc':
                ax.set_title('DRC')
            else:
                ax.set_title(location.capitalize())
    for pn in range(len(locations)):
        axes[pn].set_ylim(bottom=0)
    sc.SIticks(ax1)
    ax1.legend()
    fig.tight_layout()
    fig_name = f'{figfolder}/residual_burden.png'
    sc.savefig(fig_name, dpi=100)

    return

def plot_CEA(locations=None, background_scens=None, txvx_scen=None, discounting=False):

    set_font(size=14)
    econdfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econdf
    econ_df = pd.concat(econdfs)

    cost_dict = dict(
        txv=8,
        leep=41.76,
        ablation=11.76,
        cancer=450
    )

    standard_le = 88.8
    colors = ['orange', 'red', 'darkred']

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        dalys_noTxV = 0
        dalys_TxV = 0
        cost_noTxV = 0
        cost_TxV = 0
        for location in locations:
            NoTxV_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                   & (econ_df.txvx_scen == 'No TxV') & (econ_df.location == location)].groupby('year')[
                ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                 'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

            NoTxV_econdf_means = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                   & (econ_df.txvx_scen == 'No TxV') & (econ_df.location == location)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_counts['new_cancers'].values)])
                cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_counts['new_cancer_deaths'].values)])
            else:
                cancers = NoTxV_econdf_counts['new_cancers'].values
                cancer_deaths = NoTxV_econdf_counts['new_cancer_deaths'].values
            avg_age_ca_death = np.mean(NoTxV_econdf_means['av_age_cancer_deaths'])
            avg_age_ca = np.mean(NoTxV_econdf_means['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            daly_noTxV = yll + yld
            dalys_noTxV += daly_noTxV
            total_cost_noTxV = (NoTxV_econdf_counts['new_tx_vaccinations'].values * cost_dict['txv']) + \
                               (NoTxV_econdf_counts['new_thermal_ablations'].values * cost_dict['ablation']) + \
                               (NoTxV_econdf_counts['new_leeps'].values * cost_dict['leep']) + \
                               (NoTxV_econdf_counts['new_cancer_treatments'].values * cost_dict['cancer'])
            if discounting:
                cost_noTxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_noTxV)])
            else:
                cost_noTxV = np.sum(total_cost_noTxV)
            cost_noTxV += cost_noTxV
            txvx_scen_label_age = f'{txvx_scen}'

            TxV_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                 & (econ_df.txvx_scen == txvx_scen_label_age) & (econ_df.location == location)].groupby(
                'year')[
                ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                 'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

            TxV_econdf_means = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                        & (econ_df.txvx_scen == txvx_scen_label_age) & (
                                                    econ_df.location == location)].groupby(
                'year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_counts['new_cancers'].values)])
                cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(TxV_econdf_counts['new_cancer_deaths'].values)])
            else:
                cancers = TxV_econdf_counts['new_cancers'].values
                cancer_deaths = TxV_econdf_counts['new_cancer_deaths'].values
            avg_age_ca_death = np.mean(TxV_econdf_means['av_age_cancer_deaths'])
            avg_age_ca = np.mean(TxV_econdf_means['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            daly_TxV = yll + yld
            dalys_TxV += daly_TxV

            total_cost_TxV = (TxV_econdf_counts['new_tx_vaccinations'].values * cost_dict['txv']) + \
                             (TxV_econdf_counts['new_thermal_ablations'].values * cost_dict['ablation']) + \
                             (TxV_econdf_counts['new_leeps'].values * cost_dict['leep']) + \
                             (TxV_econdf_counts['new_cancer_treatments'].values * cost_dict['cancer'])
            if discounting:
                cost_TxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_TxV)])
            else:
                cost_TxV = np.sum(total_cost_TxV)
            cost_TxV += cost_TxV

        dalys_averted = dalys_noTxV - dalys_TxV
        additional_cost = cost_TxV - cost_noTxV
        cost_daly_averted = additional_cost / dalys_averted

        ax.plot(dalys_averted/1e6, cost_daly_averted, color=colors[ib], marker='s', linestyle = 'None', markersize=20,
                label=background_scen_label)


    # sc.SIticks(ax)
    ax.legend(title='Background intervention scenario')
    ax.set_xlabel('DALYs averted (millions), 2030-2060')
    ax.set_ylabel('Incremental costs/DALY averted,\n$USD 2030-2060')

    ax.set_ylim([0, 120])
    # fig.suptitle(f'TxV CEA for {locations}', fontsize=18)
    fig.tight_layout()
    fig_name = f'{figfolder}/CEA.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_txv_impact_combined(locations=None, background_scens=None, txvx_ages=None, txvx_efficacies=None, discounting=False):

    set_font(size=24)
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{resfolder}/{location}.obj')
        dfs += df
        econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    width = 0.1
    standard_le = 88.8

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']
    LTFU = 0.2
    fig, axes = pl.subplots(3, 1, figsize=(16, 20))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                         & (bigdf.txvx_scen == 'No TxV')].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high', 'n_tx_vaccinated']].sum()

        NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                         & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['new_cancers', 'new_cancer_deaths']].sum()

        NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                         & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()
        if discounting:
            cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
            cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
        else:
            cancers = NoTxV_econdf_cancers['new_cancers'].values
            cancer_deaths = NoTxV_econdf_cancers['new_cancer_deaths'].values

        avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
        yll = np.sum((standard_le-avg_age_ca_death) * cancer_deaths)
        daly_noTxV = yll + yld

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df['cancers'])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df['cancers_low'])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df['cancers_high'])[ys:ye])

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df['cancer_deaths'])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(np.array(NoTxV_df['cancer_deaths_low'])[ys:ye])
        NoTxV_cancer_deaths_short_high = np.sum(np.array(NoTxV_df['cancer_deaths_high'])[ys:ye])

        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            for i_age, txvx_age in enumerate(txvx_ages):
                txvx_scen_label_age = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'
                TxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label_age)].groupby(
                    'year')[
                    ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                     'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',
                     'n_tx_vaccinated']].sum()

                TxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                            & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths']].sum()

                TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                    & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                if discounting:
                    cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                    cancer_deaths = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
                else:
                    cancers = TxV_econdf_cancers['new_cancers'].values
                    cancer_deaths = TxV_econdf_cancers['new_cancer_deaths'].values

                avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                daly_TxV = yll + yld
                dalys_averted = daly_noTxV - daly_TxV

                TxV_cancers = np.sum(np.array(TxV_df['cancers'])[ys:ye])
                TxV_cancers_low = np.sum(np.array(TxV_df['cancers_low'])[ys:ye])
                TxV_cancers_high = np.sum(np.array(TxV_df['cancers_high'])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(np.array(TxV_df['cancer_deaths'])[ys:ye])
                best_TxV_cancer_deaths_short_high = np.sum(np.array(TxV_df['cancer_deaths_high'])[ys:ye])
                best_TxV_cancer_deaths_short_low = np.sum(np.array(TxV_df['cancer_deaths_low'])[ys:ye])

                averted_cancer_deaths_short = NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                averted_cancer_deaths_short_high = NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                averted_cancer_deaths_short_low = NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low

                TxV_cancers_averted = NoTxV_cancers - TxV_cancers
                TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
                TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high
                if i_eff + ib == 0:
                    axes[0].scatter(xes[i_eff][ib], TxV_cancers_averted, marker=markers[i_age], color=colors[i_eff],
                                       s=300, label=f'Age {txvx_age}')
                else:
                    axes[0].scatter(xes[i_eff][ib], TxV_cancers_averted, marker=markers[i_age], color=colors[i_eff],
                                       s=300)
                if ib + i_age == 0:
                    axes[1].scatter(xes[i_eff][ib], averted_cancer_deaths_short, marker=markers[i_age], color=colors[i_eff], s=300,
                                       label=txvx_efficacy)
                else:
                    axes[1].scatter(xes[i_eff][ib], averted_cancer_deaths_short, marker=markers[i_age], color=colors[i_eff], s=300)
                axes[0].vlines(xes[i_eff][ib], ymin=TxV_cancers_averted_low, color=colors[i_eff], ymax=TxV_cancers_averted_high)
                axes[2].scatter(xes[i_eff][ib], dalys_averted, marker=markers[i_age], color=colors[i_eff], s=300)
                axes[1].vlines(xes[i_eff][ib], ymin=averted_cancer_deaths_short_low, ymax=averted_cancer_deaths_short_high, color=colors[i_eff])

        ib_labels = background_scens.keys()
        axes[2].set_xticks([r + 1.5*width for r in range(len(r2))], ib_labels)
        axes[0].get_xaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)

    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    sc.SIticks(axes[2])

    axes[0].set_ylabel(f'Cervical cancer cases averted\n(2030-2060)')
    axes[2].set_ylabel(f'DALYs averted (2030-2060)')
    axes[1].set_ylabel(f'Cervical cancer deaths averted\n(2030-2060)')

    axes[0].legend(title='Age of TxV')
    axes[1].legend(title='TxV Effectiveness')
    axes[2].set_xlabel('Background intervention scenario')
    fig.tight_layout()
    fig_name = f'{figfolder}/txv_impact.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_txv_impact_combined_v2(locations=None, background_scens=None, txvx_ages=None, txvx_efficacies=None, discounting=False):

    set_font(size=24)
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{resfolder}/{location}.obj')
        dfs += df
        econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    width = 0.2
    standard_le = 88.8

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']

    fig = pl.figure(constrained_layout=True, figsize=(22, 16))
    spec2 = GridSpec(ncols=3, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec2[:-1, :])
    ax2 = fig.add_subplot(spec2[-1, 0])
    ax3 = fig.add_subplot(spec2[-1, 1])
    ax4 = fig.add_subplot(spec2[-1, 2])

    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                         & (bigdf.txvx_scen == 'No TxV')].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high', 'n_tx_vaccinated']].sum()

        NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                         & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['new_cancers', 'new_cancer_deaths']].sum()

        NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                         & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()

        if discounting:
            cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
            cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
        else:
            cancers = NoTxV_econdf_cancers['new_cancers'].values
            cancer_deaths = NoTxV_econdf_cancers['new_cancer_deaths'].values

        avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
        yll = np.sum((standard_le-avg_age_ca_death) * cancer_deaths)
        daly_noTxV = yll + yld

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df['cancers'])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df['cancers_low'])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df['cancers_high'])[ys:ye])

        years = np.array(NoTxV_df.index)[ys:ye]

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df['cancer_deaths'])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(np.array(NoTxV_df['cancer_deaths_low'])[ys:ye])
        NoTxV_cancer_deaths_short_high = np.sum(np.array(NoTxV_df['cancer_deaths_high'])[ys:ye])

        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            for i_age, txvx_age in enumerate(txvx_ages):
                txvx_scen_label_age = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'
                TxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label_age)].groupby(
                    'year')[
                    ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                     'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',
                     'n_tx_vaccinated']].sum()

                cancers_averted = np.array(NoTxV_df['cancers'])[ys:ye] - np.array(TxV_df['cancers'])[ys:ye]
                cancers_averted_low = np.array(NoTxV_df['cancers_low'])[ys:ye] - np.array(TxV_df['cancers_low'])[ys:ye]
                cancers_averted_high = np.array(NoTxV_df['cancers_high'])[ys:ye] - np.array(TxV_df['cancers_high'])[
                                                                                   ys:ye]
                ax1.plot(years, cancers_averted, color=colors[ib+1], linewidth=3, label=background_scen_label)
                ax1.fill_between(years, cancers_averted_low, cancers_averted_high, color=colors[ib+1], alpha=0.3)

                TxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                            & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths']].sum()

                TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                    & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                if discounting:
                    cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                    cancer_deaths = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
                else:
                    cancers = TxV_econdf_cancers['new_cancers'].values
                    cancer_deaths = TxV_econdf_cancers['new_cancer_deaths'].values

                avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                daly_TxV = yll + yld
                dalys_averted = daly_noTxV - daly_TxV

                TxV_cancers = np.sum(np.array(TxV_df['cancers'])[ys:ye])
                TxV_cancers_low = np.sum(np.array(TxV_df['cancers_low'])[ys:ye])
                TxV_cancers_high = np.sum(np.array(TxV_df['cancers_high'])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(np.array(TxV_df['cancer_deaths'])[ys:ye])
                best_TxV_cancer_deaths_short_high = np.sum(np.array(TxV_df['cancer_deaths_high'])[ys:ye])
                best_TxV_cancer_deaths_short_low = np.sum(np.array(TxV_df['cancer_deaths_low'])[ys:ye])

                averted_cancer_deaths_short = NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                averted_cancer_deaths_short_high = NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                averted_cancer_deaths_short_low = NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low

                TxV_cancers_averted = NoTxV_cancers - TxV_cancers
                TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
                TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high
                if i_eff + ib == 0:
                    ax2.scatter(xes[i_eff][ib], TxV_cancers_averted, marker=markers[i_age], color='b',
                                       s=300, label=f'{txvx_age}')
                else:
                    ax2.scatter(xes[i_eff][ib], TxV_cancers_averted, marker=markers[i_age], color='b',
                                       s=300)
                ax3.scatter(xes[i_eff][ib], averted_cancer_deaths_short, marker=markers[i_age], color='b', s=300)
                ax2.vlines(xes[i_eff][ib], ymin=TxV_cancers_averted_low, color='b', ymax=TxV_cancers_averted_high)
                ax4.scatter(xes[i_eff][ib], dalys_averted, marker=markers[i_age], color='b', s=300)
                ax3.vlines(xes[i_eff][ib], ymin=averted_cancer_deaths_short_low, ymax=averted_cancer_deaths_short_high, color='b')

        ib_labels = background_scens.keys()
        ax2.set_xticks([r + 1.5*width for r in range(len(r2))], ib_labels)
        ax3.set_xticks([r + 1.5 * width for r in range(len(r2))], ib_labels)
        ax4.set_xticks([r + 1.5 * width for r in range(len(r2))], ib_labels)
        # axes[0].get_xaxis().set_visible(False)
        # axes[1].get_xaxis().set_visible(False)

    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)
    sc.SIticks(ax1)
    sc.SIticks(ax2)
    sc.SIticks(ax3)
    sc.SIticks(ax4)

    ax1.set_ylabel(f'Cervical cancer cases averted')
    ax2.set_ylabel(f'Cervical cancer cases averted (2030-2060)')
    ax3.set_ylabel(f'Cervical cancer deaths averted (2030-2060)')
    ax4.set_ylabel(f'DALYs averted (2030-2060)')

    # ax2.legend(title='Age of TxV')
    ax1.legend(title='Background intervention scale-up')
    ax2.set_xlabel('Background intervention scenario')
    ax3.set_xlabel('Background intervention scenario')
    ax4.set_xlabel('Background intervention scenario')
    fig.tight_layout()
    fig_name = f'{figfolder}/txv_impactv2.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_txv_impact_comparison(locations=None, background_scens=None, txvx_ages=None, txvx_efficacies=None,
                               do_run=True, discounting=False):

    if do_run:
        dfs = sc.autolist()
        for location in locations:
            dt_dfs = sc.autolist()
            dt = an.dwelltime_by_genotype(start_year=2000)
            calib_par_stem = '_nov13'
            calib_pars = sc.loadobj(f'results/{location}_pars{calib_par_stem}.obj')
            sim = rs.run_sim(location=location, calib_pars=calib_pars, end=2020, n_agents=50e3, analyzers=[dt])
            dt_res = sim.analyzers[0]
            dt_df = pd.DataFrame()
            dt_df['Age'] = dt_res.age_causal
            dt_df['Health event'] = 'Infection'
            dt_df['location'] = location
            dt_dfs += dt_df

            dt_df = pd.DataFrame()
            dt_df['Age'] = dt_res.age_cin
            dt_df['Health event'] = 'CIN'
            dt_df['location'] = location
            dt_dfs += dt_df

            dt_df = pd.DataFrame()
            dt_df['Age'] = dt_res.age_cancer
            dt_df['Health event'] = 'Cancer'
            dt_df['location'] = location
            dt_dfs += dt_df
            df = pd.concat(dt_dfs)
            dfs += df

    set_font(size=24)

    econdfs = sc.autolist()
    for location in locations:
        econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    colors = sc.gridcolors(20)
    width = 0.2
    standard_le = 88.8

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']

    fig, axes = pl.subplots(ncols=2, nrows=2, sharey='row', figsize=(16, 16))

    for iloc, location in enumerate(locations):

        sns.violinplot(x="Health event",
                           y="Age",
                           data=dfs[iloc], ax=axes[0,iloc])
        for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
            vx_scen_label = background_scen['vx_scen']
            screen_scen_label = background_scen['screen_scen']
            NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                             & (econdf.txvx_scen == 'No TxV') & (econdf.location == location)].groupby('year')[
                ['new_cancers', 'new_cancer_deaths']].sum()

            NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                             & (econdf.txvx_scen == 'No TxV') & (econdf.location == location)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
                cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
            else:
                cancers = NoTxV_econdf_cancers['new_cancers'].values
                cancer_deaths = NoTxV_econdf_cancers['new_cancer_deaths'].values

            avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
            avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le-avg_age_ca_death) * cancer_deaths)
            daly_noTxV = yll + yld


            for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
                for i_age, txvx_age in enumerate(txvx_ages):
                    txvx_scen_label_age = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'


                    TxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                                & (econdf.txvx_scen == txvx_scen_label_age) & (econdf.location == location)].groupby('year')[
                        ['new_cancers', 'new_cancer_deaths']].sum()

                    TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                        & (econdf.txvx_scen == txvx_scen_label_age) & (econdf.location == location)].groupby('year')[
                        ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                    if discounting:
                        cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                        cancer_deaths = np.array(
                            [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
                    else:
                        cancers = TxV_econdf_cancers['new_cancers'].values
                        cancer_deaths = TxV_econdf_cancers['new_cancer_deaths'].values

                    avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                    avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                    ca_years = avg_age_ca_death - avg_age_ca
                    yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                    yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                    daly_TxV = yll + yld
                    dalys_averted = daly_noTxV - daly_TxV
                    perc_dalys_averted = 100*dalys_averted/daly_noTxV

                    if iloc + i_eff + ib == 0:
                        axes[1,iloc].scatter(xes[i_eff][ib], perc_dalys_averted, marker=markers[i_age], color=colors[i_eff],
                                           s=300, label=f'{txvx_age}')
                    elif iloc == 1 and i_age == 0 and ib == 0:
                        axes[1,iloc].scatter(xes[i_eff][ib], perc_dalys_averted, marker=markers[i_age],
                                              color=colors[i_eff], s=300, label=txvx_efficacy)
                    else:
                        axes[1,iloc].scatter(xes[i_eff][ib], perc_dalys_averted, marker=markers[i_age],
                                              color=colors[i_eff], s=300)

            ib_labels = background_scens.keys()
            axes[1,iloc].set_xticks([r + 1.5*width for r in range(len(r2))], ib_labels)
            axes[0,iloc].set_title(location.capitalize())


            sc.SIticks(axes[1,iloc])


    axes[1,0].set_ylabel(f'Percent DALYs averted (2030-2060)')

    axes[1,0].legend(title='Age of TxV')
    axes[1,1].legend(title='TxV Effectiveness', ncol=2 )
    axes[1,0].set_xlabel('Background intervention scenario')
    axes[1,1].set_xlabel('Background intervention scenario')
    axes[1,0].set_ylim(0,30)
    axes[1,1 ].set_ylim(0,30)
    fig.tight_layout()
    fig_name = f'{figfolder}/txv_impact_country_comparison.png'
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_age(locations=None, background_scens=None, txvx_scens=None):

    set_font(size=24)
    econdfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econdf
    econ_df = pd.concat(econdfs)

    cost_dict = dict(
        txv=8,
        leep=41.76,
        ablation=11.76,
        cancer=450
    )

    standard_le = 88.8
    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']
    colors = ['gold', 'orange', 'red', 'darkred']

    handles = []

    fig, ax = pl.subplots(figsize=(12, 12))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        for it, txvx_scen in enumerate(txvx_scens):
            dalys_noTxV = 0
            dalys_TxV = 0
            cost_noTxV = 0
            cost_TxV = 0
            for location in locations:

                NoTxV_econdf = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                 & (econ_df.txvx_scen == 'No TxV') & (econ_df.location == location)].groupby('year')[
                    ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                     'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths', 'av_age_cancer_deaths', 'av_age_cancers']].sum()

                discounted_cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf['new_cancers'].values)])
                discounted_cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf['new_cancer_deaths'].values)])
                avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
                yll = np.sum((standard_le-avg_age_ca_death) * discounted_cancer_deaths)
                daly_noTxV = yll + yld
                dalys_noTxV += daly_noTxV
                total_cost_noTxV = (NoTxV_econdf['new_tx_vaccinations'].values * cost_dict['txv']) + \
                             (NoTxV_econdf['new_thermal_ablations'].values * cost_dict['ablation']) + \
                             (NoTxV_econdf['new_leeps'].values * cost_dict['leep']) + \
                             (NoTxV_econdf['new_cancers'].values * cost_dict['cancer'])
                discounted_cost_noTxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_noTxV)])
                cost_noTxV += discounted_cost_noTxV
                txvx_scen_label_age = f'{txvx_scen}'

                TxV_econdf = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                      & (econ_df.txvx_scen == txvx_scen_label_age) & (econ_df.location == location)].groupby('year')[
                    ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                     'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths', 'av_age_cancer_deaths', 'av_age_cancers']].sum()

                discounted_cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf['new_cancers'].values)])
                discounted_cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(TxV_econdf['new_cancer_deaths'].values)])
                avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * discounted_cancer_deaths)
                daly_TxV = yll + yld
                dalys_TxV += daly_TxV

                total_cost_TxV = (TxV_econdf['new_tx_vaccinations'].values * cost_dict['txv']) + \
                                   (TxV_econdf['new_thermal_ablations'].values * cost_dict['ablation']) + \
                                   (TxV_econdf['new_leeps'].values * cost_dict['leep']) + \
                                   (TxV_econdf['new_cancer_treatments'].values * cost_dict['cancer'])
                discounted_cost_TxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_TxV)])
                cost_TxV += discounted_cost_TxV

            dalys_averted = dalys_noTxV - dalys_TxV
            additional_cost = cost_TxV - cost_noTxV
            cost_daly_averted = additional_cost / dalys_averted
            handle, = ax.plot(dalys_averted/1e6, cost_daly_averted, marker=markers[ib], linestyle = 'None',
                               color=colors[it], markersize=20)
            handles.append(handle)

    # sc.SIticks(ax)
    # legend1 = ax.legend(handles[0:3], ["30", "35", "40"], bbox_to_anchor= (1.0, 0.5), title='Age of TxV')
    # ax.legend([handles[0], handles[3], handles[6], handles[9]], background_scens.keys(), loc='upper right', title='Background intervention scale-up')
    # pl.gca().add_artist(legend1)
    ax.set_xlabel('DALYs averted (millions), 2030-2060')
    ax.set_ylabel('Incremental costs/DALY averted, $USD 2030-2060')
    # fig.suptitle(f'TxV CEA for {locations}', fontsize=18)
    fig.tight_layout()
    fig_name = f'{figfolder}/CEA_age.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_txv_impact_combined_age(locations=None, background_scens=None, txvx_scens=None):

    set_font(size=24)
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{resfolder}/{location}.obj')
        dfs += df
        econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    bigdf = pd.concat(dfs)
    colors = ['gold', 'orange', 'red', 'darkred']
    width = 0.15
    standard_le = 88.8

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    # r4 = [x + width for x in r3]
    xes = [r1, r2, r3]#, r4]
    LTFU = 0.2
    fig, axes = pl.subplots(2, 2, figsize=(16, 16))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                         & (bigdf.txvx_scen == 'No TxV')].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high', 'n_tx_vaccinated']].sum()

        NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                         & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['new_cancers', 'new_cancer_deaths']].sum()

        NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                         & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()

        discounted_cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
        discounted_cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
        avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
        yll = np.sum((standard_le-avg_age_ca_death) * discounted_cancer_deaths)
        daly_noTxV = yll + yld

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df['cancers'])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df['cancers_low'])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df['cancers_high'])[ys:ye])

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df['cancer_deaths'])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(np.array(NoTxV_df['cancer_deaths_low'])[ys:ye])
        NoTxV_cancer_deaths_short_high = np.sum(np.array(NoTxV_df['cancer_deaths_high'])[ys:ye])

        for it, txvx_scen in enumerate(txvx_scens):
            txvx_scen_label_age = f'{txvx_scen}'
            TxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                           & (bigdf.txvx_scen == txvx_scen_label_age)].groupby(
                'year')[
                ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                 'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high', 'n_tx_vaccinated']].sum()

            TxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                          & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                ['new_cancers', 'new_cancer_deaths']].sum()

            TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                  & (econdf.txvx_scen == txvx_scen_label_age)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            discounted_cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
            discounted_cancer_deaths = np.array(
                [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
            avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
            avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * discounted_cancer_deaths)
            daly_TxV = yll + yld
            dalys_averted = daly_noTxV - daly_TxV
            best_TxV_cancers_short = np.sum(np.array(TxV_df['cancers'])[ys:ye])
            averted_cancers_short = NoTxV_cancers - best_TxV_cancers_short
            perc_cancers_averted_short = 100 * averted_cancers_short / NoTxV_cancers
            to_plot_short = perc_cancers_averted_short

            TxV_cancers = np.sum(np.array(TxV_df['cancers'])[ys:ye])
            TxV_cancers_low = np.sum(np.array(TxV_df['cancers_low'])[ys:ye])
            TxV_cancers_high = np.sum(np.array(TxV_df['cancers_high'])[ys:ye])

            best_TxV_cancer_deaths_short = np.sum(np.array(TxV_df['cancer_deaths'])[ys:ye])
            best_TxV_cancer_deaths_short_high = np.sum(np.array(TxV_df['cancer_deaths_high'])[ys:ye])
            best_TxV_cancer_deaths_short_low = np.sum(np.array(TxV_df['cancer_deaths_low'])[ys:ye])

            averted_cancer_deaths_short = NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
            averted_cancer_deaths_short_high = NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
            averted_cancer_deaths_short_low = NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low

            yev = sc.findinds(NoTxV_df.index, 2050)[0]
            n_vaccinated = np.array(TxV_df['n_tx_vaccinated'])[yev]
            NNV = 1000 * (averted_cancer_deaths_short / ((1 - LTFU) * n_vaccinated))
            NNV_low = 1000 * (averted_cancer_deaths_short_low / ((1 - LTFU) * n_vaccinated))
            NNV_high = 1000 * (averted_cancer_deaths_short_high / ((1 - LTFU) * n_vaccinated))

            NNV_dalys = 1000 * (dalys_averted / ((1 - LTFU) * n_vaccinated))

            years = np.array(TxV_df.index)[ys:ye]
            TxV_cancers_averted = NoTxV_cancers - TxV_cancers
            TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
            TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high
            if ib == 0:
                label = txvx_scen.replace('Mass TxV, 90/50, ', '')
                axes[0, 0].bar(xes[it][ib], TxV_cancers_averted, width=width, color=colors[it], label=label)
            axes[0,0].bar(xes[it][ib], TxV_cancers_averted, width=width, color=colors[it])
            axes[0,1].bar(xes[it][ib], dalys_averted, width=width, color=colors[it])
            axes[1,0].scatter(xes[it][ib], NNV, marker="s", color=colors[it], s=300)
            axes[1,0].vlines(xes[it][ib], ymin=NNV_low, ymax=NNV_high, color=colors[it])
            axes[1,1].scatter(xes[it][ib], NNV_dalys, marker="s", color=colors[it], s=300)

        ib_labels = background_scens.keys()

        axes[1,0].set_xticks([r + 1.5*width for r in range(len(r1))], ib_labels)
        axes[1,1].set_xticks([r + 1.5 * width for r in range(len(r1))], ib_labels)

    sc.SIticks(axes[0,0])
    sc.SIticks(axes[0,1])
    sc.SIticks(axes[1, 0])
    sc.SIticks(axes[1,1])

    axes[0,0].set_ylabel(f'Cervical cancer cases averted (2030-2060)')
    axes[0,1].set_ylabel(f'DALYs averted (2030-2060)')
    axes[1,0].set_ylabel(f'Deaths averted per 1,000 FVPs')
    axes[1,1].set_ylabel(f'DALYs averted per 1,000 FVPs')

    # axes[1].set_ylim([0, 30])
    axes[0,0].legend(title='TxV Target Age')

    # axes[1].set_xlabel('Background intervention scenario')
    axes[1,0].set_xlabel('Background intervention scenario')
    axes[1, 1].set_xlabel('Background intervention scenario')
    fig.tight_layout()
    fig_name = f'{figfolder}/txv_impact.png'
    fig.savefig(fig_name, dpi=100)

    return

def compile_IPM_data(locations=None, background_scens=None, txvx_scen=None):
    econdfs = sc.autolist()
    for location in locations:
        econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econ_df

    econdf = pd.concat(econdfs)

    ipm_dfs = sc.autolist()
    standard_le = 88.8

    cost_dict = dict(
        txv=8,
        leep=41.76,
        ablation=11.76,
        cancer=450
    )
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']

        for location in locations:

            NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                   & (econdf.txvx_scen == 'No TxV') & (econdf.location == location)].groupby('year')[
                ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                 'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths', 'av_age_cancer_deaths',
                 'av_age_cancers']].sum()

            cancers = NoTxV_econdf['new_cancers'].values
            cancer_deaths = NoTxV_econdf['new_cancer_deaths'].values
            avg_age_ca_death = NoTxV_econdf['av_age_cancer_deaths'].values
            avg_age_ca = NoTxV_econdf['av_age_cancers'].values
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers
            yll = (standard_le - avg_age_ca_death) * cancer_deaths
            daly_noTxV = yll + yld
            total_cost_noTxV = (NoTxV_econdf['new_tx_vaccinations'].values * cost_dict['txv']) + \
                               (NoTxV_econdf['new_thermal_ablations'].values * cost_dict['ablation']) + \
                               (NoTxV_econdf['new_leeps'].values * cost_dict['leep']) + \
                               (NoTxV_econdf['new_cancers'].values * cost_dict['cancer'])

            ipm_df = pd.DataFrame()
            ipm_df['years'] = np.arange(2020, 2061)
            ipm_df['location'] = location
            ipm_df['dalys'] = daly_noTxV
            ipm_df['costs'] = total_cost_noTxV
            ipm_df['txv_doses'] = NoTxV_econdf['new_tx_vaccinations'].values
            ipm_df['leeps'] = NoTxV_econdf['new_leeps'].values
            ipm_df['thermal_ablations'] = NoTxV_econdf['new_thermal_ablations'].values
            ipm_df['cancer_treatments'] = NoTxV_econdf['new_cancer_treatments'].values
            ipm_df['cancers'] = cancers
            ipm_df['cancer_deaths'] = cancer_deaths
            ipm_df['PxV'] = vx_scen_label
            ipm_df['S&T'] = screen_scen_label
            ipm_df['TxV'] = 'No TxV'
            ipm_dfs += ipm_df
            txvx_scen_label_age = f'{txvx_scen}'
            TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                 & (econdf.txvx_scen == txvx_scen_label_age) & (econdf.location == location)].groupby(
                'year')[
                ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                 'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths', 'av_age_cancer_deaths',
                 'av_age_cancers']].sum()

            cancers = TxV_econdf['new_cancers'].values
            cancer_deaths = TxV_econdf['new_cancer_deaths'].values
            avg_age_ca_death = TxV_econdf['av_age_cancer_deaths'].values
            avg_age_ca = TxV_econdf['av_age_cancers'].values
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers
            yll = (standard_le - avg_age_ca_death) * cancer_deaths
            daly_TxV = yll + yld
            total_cost_TxV = (TxV_econdf['new_tx_vaccinations'].values * cost_dict['txv']) + \
                             (TxV_econdf['new_thermal_ablations'].values * cost_dict['ablation']) + \
                             (TxV_econdf['new_leeps'].values * cost_dict['leep']) + \
                             (TxV_econdf['new_cancer_treatments'].values * cost_dict['cancer'])

            ipm_df = pd.DataFrame()
            ipm_df['years'] = np.arange(2020, 2061)
            ipm_df['location'] = location
            ipm_df['dalys'] = daly_TxV
            ipm_df['costs'] = total_cost_TxV
            ipm_df['txv_doses'] = TxV_econdf['new_tx_vaccinations'].values
            ipm_df['leeps'] = TxV_econdf['new_leeps'].values
            ipm_df['thermal_ablations'] = TxV_econdf['new_thermal_ablations'].values
            ipm_df['cancer_treatments'] = TxV_econdf['new_cancer_treatments'].values
            ipm_df['cancers'] = cancers
            ipm_df['cancer_deaths'] = cancer_deaths
            ipm_df['PxV'] = vx_scen_label
            ipm_df['S&T'] = screen_scen_label
            ipm_df['TxV'] = txvx_scen_label_age
            ipm_dfs += ipm_df

    final_df = pd.concat(ipm_dfs)
    filename = f'{resfolder}/IPM_compare.csv'
    final_df.to_csv(filename)
    return

def make_sens(location=None, background_scens=None, txvx_efficacies=None, txvx_ages=None, sensitivities=None):
    sens_labels = {
        '': ['Baseline', 'baseline'],
        ', cross-protection': ['50% cross-protection', 'cross_protection'],
        ', 0.05 decay': ['5% annual decay since virus/lesion', '0.05decay'],
        ', intro 2035':['2035 introduction', 'intro_2035'],
        ', no durable immunity': ['No immune memory']
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    set_font(size=24)

    bigdf = sc.loadobj(f'{resfolder}/{location}.obj')
    econdf = sc.loadobj(f'{resfolder}/{location}_econ.obj')

    colors = sc.gridcolors(20)
    width = 0.1
    standard_le = 88.8

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]
    sizes = [500, 300, 100]
    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']
    fig, axes = pl.subplots(3, 1, figsize=(16, 20))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                         & (bigdf.txvx_scen == 'No TxV')].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',
             'n_tx_vaccinated']].sum()

        NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                      & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['new_cancers', 'new_cancer_deaths']].sum()

        NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                              & (econdf.txvx_scen == 'No TxV')].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()

        discounted_cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
        discounted_cancer_deaths = np.array(
            [i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
        avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
        yll = np.sum((standard_le - avg_age_ca_death) * discounted_cancer_deaths)
        daly_noTxV = yll + yld

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]


        NoTxV_cancers = np.sum(np.array(NoTxV_df['cancers'])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df['cancers_low'])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df['cancers_high'])[ys:ye])

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df['cancer_deaths'])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(np.array(NoTxV_df['cancer_deaths_low'])[ys:ye])
        NoTxV_cancer_deaths_short_high = np.sum(np.array(NoTxV_df['cancer_deaths_high'])[ys:ye])

        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            for i_age, txvx_age in enumerate(txvx_ages):
                txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'
                baseline_TxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label)].groupby(
                    'year')[
                    ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                     'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high',
                     'n_tx_vaccinated']].sum()

                baseline_TxV_econdf_cancers = \
                    econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                           & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
                        ['new_cancers', 'new_cancer_deaths']].sum()

                baseline_TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                    & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                discounted_cancers = np.array(
                    [i / 1.03 ** t for t, i in enumerate(baseline_TxV_econdf_cancers['new_cancers'].values)])
                discounted_cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(baseline_TxV_econdf_cancers['new_cancer_deaths'].values)])
                avg_age_ca_death = np.mean(baseline_TxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(baseline_TxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * discounted_cancer_deaths)
                daly_TxV = yll + yld
                dalys_averted_baseline = daly_noTxV - daly_TxV

                TxV_cancers = np.sum(np.array(baseline_TxV_df['cancers'])[ys:ye])
                TxV_cancers_low = np.sum(np.array(baseline_TxV_df['cancers_low'])[ys:ye])
                TxV_cancers_high = np.sum(np.array(baseline_TxV_df['cancers_high'])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(np.array(baseline_TxV_df['cancer_deaths'])[ys:ye])
                best_TxV_cancer_deaths_short_high = np.sum(np.array(baseline_TxV_df['cancer_deaths_high'])[ys:ye])
                best_TxV_cancer_deaths_short_low = np.sum(np.array(baseline_TxV_df['cancer_deaths_low'])[ys:ye])

                averted_cancer_deaths_short_baseline = NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                averted_cancer_deaths_short_baseline_high = NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                averted_cancer_deaths_short_baseline_low = NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low

                TxV_cancers_averted_baseline = NoTxV_cancers - TxV_cancers
                TxV_cancers_averted_baseline_low = NoTxV_cancers_low - TxV_cancers_low
                TxV_cancers_averted_baseline_high = NoTxV_cancers_high - TxV_cancers_high
                for isens, sens_label in enumerate(sensitivities):
                    txvx_scen_label_sen = f'{txvx_scen_label}{sens_label}'
                    TxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                                   & (bigdf.txvx_scen == txvx_scen_label_sen)].groupby(
                        'year')[
                        ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                         'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                         'cancer_deaths_high',
                         'n_tx_vaccinated']].sum()

                    TxV_econdf_cancers = \
                    econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                           & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                        ['new_cancers', 'new_cancer_deaths']].sum()

                    TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                        & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                        ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                    discounted_cancers = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                    discounted_cancer_deaths = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
                    avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                    avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                    ca_years = avg_age_ca_death - avg_age_ca
                    yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * discounted_cancers)
                    yll = np.sum((standard_le - avg_age_ca_death) * discounted_cancer_deaths)
                    daly_TxV = yll + yld
                    dalys_averted = daly_noTxV - daly_TxV

                    TxV_cancers = np.sum(np.array(TxV_df['cancers'])[ys:ye])
                    TxV_cancers_low = np.sum(np.array(TxV_df['cancers_low'])[ys:ye])
                    TxV_cancers_high = np.sum(np.array(TxV_df['cancers_high'])[ys:ye])

                    best_TxV_cancer_deaths_short = np.sum(np.array(TxV_df['cancer_deaths'])[ys:ye])
                    best_TxV_cancer_deaths_short_high = np.sum(np.array(TxV_df['cancer_deaths_high'])[ys:ye])
                    best_TxV_cancer_deaths_short_low = np.sum(np.array(TxV_df['cancer_deaths_low'])[ys:ye])

                    averted_cancer_deaths_short = NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                    averted_cancer_deaths_short_high = NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                    averted_cancer_deaths_short_low = NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low

                    TxV_cancers_averted = NoTxV_cancers - TxV_cancers
                    TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
                    TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high
                    if ib + i_age + i_eff  == 0:
                        axes[0].scatter(xes[i_eff][ib], TxV_cancers_averted-TxV_cancers_averted_baseline, marker=markers[i_age], color=colors[i_eff],
                                        s=sizes[isens], label=f'{sens_labels_to_use[isens]}')
                    else:
                        axes[0].scatter(xes[i_eff][ib], TxV_cancers_averted-TxV_cancers_averted_baseline, marker=markers[i_age], color=colors[i_eff],
                                        s=sizes[isens])
                    if i_eff + ib + isens == 0:

                        axes[1].scatter(xes[i_eff][ib], averted_cancer_deaths_short-averted_cancer_deaths_short_baseline, marker=markers[i_age],
                                        color=colors[i_eff], s=sizes[isens],
                                        label=txvx_age)
                    else:
                        axes[1].scatter(xes[i_eff][ib], averted_cancer_deaths_short-averted_cancer_deaths_short_baseline, marker=markers[i_age],
                                        color=colors[i_eff], s=sizes[isens])
                    if ib + isens + i_age == 0:
                        label = txvx_scen_label.replace('Mass TxV, ', '')
                        label = label.replace(', age 35', '')
                        axes[2].scatter(xes[i_eff][ib], dalys_averted-dalys_averted_baseline, marker=markers[i_age],
                                        color=colors[i_eff], s=sizes[isens], label=label)
                    else:
                        axes[2].scatter(xes[i_eff][ib], dalys_averted - dalys_averted_baseline, marker=markers[i_age],
                                        color=colors[i_eff], s=sizes[isens])

        ib_labels = background_scens.keys()
        axes[2].set_xticks([r + 1.5 * width for r in range(len(r2))], ib_labels)
        axes[0].get_xaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)

    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    sc.SIticks(axes[2])

    axes[0].set_ylabel(f'Cervical cancer cases averted\n(relative to baseline)')
    axes[2].set_ylabel(f'DALYs averted\n(relative to baseline)')
    axes[1].set_ylabel(f'Cervical cancer deaths averted\n(relative to baseline)')
    axes[0].axhline(0, color='black')
    axes[1].axhline(0, color='black')
    axes[2].axhline(0, color='black')

    axes[1].legend(title='Age of TxV', bbox_to_anchor=(1.05,0.5))
    axes[0].legend(title='Sensitivity', bbox_to_anchor=(1.05,0.5))
    axes[2].legend(title='TxV Effectiveness', bbox_to_anchor=(1.05,0.5))
    axes[2].set_xlabel('Background intervention scenario')
    fig.tight_layout()

    fig_name = f'{figfolder}/{location}_sens.png'
    fig.savefig(fig_name, dpi=100)


    return

def make_sens_combined(locations=None, discounting=False, background_scen=None, txvx_efficacy=None, txvx_ages=None, sensitivities=None):

    sens_labels = {
        '': ['Baseline', 'baseline'],
        ', cross-protection': ['50% cross-protection', '50%\ncross-protection'],
        ', 0.05 decay': ['5% annual decay since virus/lesion', '0.05decay'],
        ', intro 2035':['2035 introduction', '2035\nintroduction'],
        ', no durable immunity': ['No immune memory', 'No immune\nmemory']
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    set_font(size=20)
    econdfs = sc.autolist()
    for location in locations:
        econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    color = sc.gridcolors(20)[3]
    width = 0.2
    standard_le = 88.8

    r1 = np.arange(len(sensitivities))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']
    fig = pl.figure(constrained_layout=True, figsize=(16, 16))
    spec2 = GridSpec(ncols=3, nrows=4, figure=fig)
    ax = fig.add_subplot(spec2[0, :])
    ax1 = fig.add_subplot(spec2[1, 0])
    ax2 = fig.add_subplot(spec2[1, 1])
    ax3 = fig.add_subplot(spec2[1, 2])
    ax4 = fig.add_subplot(spec2[2, 0])
    ax5 = fig.add_subplot(spec2[2, 1])
    ax6 = fig.add_subplot(spec2[2, 2])
    ax7 = fig.add_subplot(spec2[3, 0])
    ax8 = fig.add_subplot(spec2[3, 1])
    ax9 = fig.add_subplot(spec2[3, 2])
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                  & (econdf.txvx_scen == 'No TxV')].groupby('year')[
        ['new_cancers', 'new_cancer_deaths']].sum()

    NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                          & (econdf.txvx_scen == 'No TxV')].groupby('year')[
        ['av_age_cancer_deaths', 'av_age_cancers']].mean()

    if discounting:
        cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
        cancer_deaths = np.array(
            [i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
    else:
        cancers = NoTxV_econdf_cancers['new_cancers'].values
        cancer_deaths = NoTxV_econdf_cancers['new_cancer_deaths'].values
    avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
    avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
    ca_years = avg_age_ca_death - avg_age_ca
    yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
    yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
    daly_noTxV = yll + yld

    for i_age, txvx_age in enumerate(txvx_ages):
        txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'

        baseline_TxV_econdf_cancers = \
            econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                   & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
                ['new_cancers', 'new_cancer_deaths']].sum()

        baseline_TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                     & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()
        if discounting:
            cancers = np.array(
                [i / 1.03 ** t for t, i in enumerate(baseline_TxV_econdf_cancers['new_cancers'].values)])
            cancer_deaths = np.array(
                [i / 1.03 ** t for t, i in enumerate(baseline_TxV_econdf_cancers['new_cancer_deaths'].values)])
        else:
            cancers = baseline_TxV_econdf_cancers['new_cancers'].values
            cancer_deaths = baseline_TxV_econdf_cancers['new_cancer_deaths'].values

        avg_age_ca_death = np.mean(baseline_TxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(baseline_TxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
        yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
        daly_TxV = yll + yld
        dalys_averted_baseline = daly_noTxV - daly_TxV

        for isens, sens_label in enumerate(sensitivities):
            txvx_scen_label_sen = f'{txvx_scen_label}{sens_label}'

            TxV_econdf_cancers = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths']].sum()

            TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array(
                    [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
            else:
                cancers = TxV_econdf_cancers['new_cancers'].values
                cancer_deaths = TxV_econdf_cancers['new_cancer_deaths'].values

            avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
            avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            daly_TxV = yll + yld
            dalys_averted = daly_noTxV - daly_TxV

            if isens == 0:
                ax.scatter(xes[i_age][isens], dalys_averted - dalys_averted_baseline, marker=markers[i_age],
                           color=color, s=300, label=txvx_age)
            else:
                ax.scatter(xes[i_age][isens], dalys_averted - dalys_averted_baseline, marker=markers[i_age],
                           color=color, s=300)

    ax.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use)

    for iloc, location in enumerate(locations):
        if location == 'drc':
            axes[iloc].set_title('DRC')
        else:
            axes[iloc].set_title(location.capitalize())
        NoTxV_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                      & (econdf.txvx_scen == 'No TxV') & (econdf.location == location)].groupby('year')[
            ['new_cancers', 'new_cancer_deaths']].sum()

        NoTxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                              & (econdf.txvx_scen == 'No TxV') & (econdf.location == location)].groupby('year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()
        if discounting:
            cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancers'].values)])
            cancer_deaths = np.array(
                [i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_cancers['new_cancer_deaths'].values)])
        else:
            cancers = NoTxV_econdf_cancers['new_cancers'].values
            cancer_deaths = NoTxV_econdf_cancers['new_cancer_deaths'].values

        avg_age_ca_death = np.mean(NoTxV_econdf['av_age_cancer_deaths'])
        avg_age_ca = np.mean(NoTxV_econdf['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
        yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
        daly_noTxV = yll + yld

        for i_age, txvx_age in enumerate(txvx_ages):
            txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'

            baseline_TxV_econdf_cancers = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label) & (econdf.location == location)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths']].sum()

            baseline_TxV_econdf = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label) & (econdf.location == location)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array(
                    [i / 1.03 ** t for t, i in enumerate(baseline_TxV_econdf_cancers['new_cancers'].values)])
                cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(baseline_TxV_econdf_cancers['new_cancer_deaths'].values)])
            else:
                cancers = baseline_TxV_econdf_cancers['new_cancers'].values
                cancer_deaths = baseline_TxV_econdf_cancers['new_cancer_deaths'].values

            avg_age_ca_death = np.mean(baseline_TxV_econdf['av_age_cancer_deaths'])
            avg_age_ca = np.mean(baseline_TxV_econdf['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            daly_TxV = yll + yld
            dalys_averted_baseline = daly_noTxV - daly_TxV

            for isens, sens_label in enumerate(sensitivities):
                txvx_scen_label_sen = f'{txvx_scen_label}{sens_label}'

                TxV_econdf_cancers = \
                    econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                           & (econdf.txvx_scen == txvx_scen_label_sen) & (econdf.location == location)].groupby('year')[
                        ['new_cancers', 'new_cancer_deaths']].sum()

                TxV_econdf = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                    & (econdf.txvx_scen == txvx_scen_label_sen) & (
                                                econdf.location == location)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                if discounting:
                    cancers = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancers'].values)])
                    cancer_deaths = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_cancers['new_cancer_deaths'].values)])
                else:
                    cancers = TxV_econdf_cancers['new_cancers'].values
                    cancer_deaths = TxV_econdf_cancers['new_cancer_deaths'].values

                avg_age_ca_death = np.mean(TxV_econdf['av_age_cancer_deaths'])
                avg_age_ca = np.mean(TxV_econdf['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(
                    np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                daly_TxV = yll + yld
                dalys_averted = daly_noTxV - daly_TxV
                perc_dalys_averted = 100 * (dalys_averted - dalys_averted_baseline) / dalys_averted_baseline

                axes[iloc].scatter(xes[i_age][isens], perc_dalys_averted, marker=markers[i_age],
                                       color=color, s=150)

    sens_labels_to_use = [sens_labels[sens_label][1] for sens_label in sensitivities]
    ax7.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use, fontsize=16)
    ax8.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use, fontsize=16)
    ax9.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use, fontsize=16)
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax5.get_xaxis().set_visible(False)
    ax6.get_xaxis().set_visible(False)
    sc.SIticks(ax)
    for ax_to_adjust in axes:
        sc.SIticks(ax_to_adjust)
        ax_to_adjust.axhline(0, color='black')
        ax_to_adjust.set_ylim(-75,35)

    ax.set_ylabel(f'DALYs averted\n(relative to baseline)')
    ax1.set_ylabel(f'Percent DALYs averted\n(relative to baseline)')
    ax4.set_ylabel(f'Percent DALYs averted\n(relative to baseline)')
    ax7.set_ylabel(f'Percent DALYs averted\n(relative to baseline)')
    ax.axhline(0, color='black')

    ax.legend(title='Age of TxV')
    fig.tight_layout()

    fig_name = f'{figfolder}/sens.png'
    fig.savefig(fig_name, dpi=100)


    return


def plot_natural_history(locations, do_run=True):
    if do_run:
        dfs = sc.autolist()
        for location in locations:
            dt_dfs = sc.autolist()
            dt = an.dwelltime_by_genotype(start_year=2000)
            if location in ['tanzania', 'myanmar']:
                calib_filestem = '_nov07'
            elif location in ['uganda', 'drc', 'uganda']:
                calib_filestem = '_nov06'
            else:
                calib_filestem = '_nov13'
            calib_pars = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')
            sim = rs.run_sim(location=location, calib_pars=calib_pars, end=2020, n_agents=50e3, analyzers=[dt])
            dt_res = sim.analyzers[0]
            dt_df = pd.DataFrame()
            dt_df['Age'] = dt_res.age_causal
            dt_df['Health event'] = 'Infection'
            dt_df['location'] = location
            dt_dfs += dt_df

            dt_df = pd.DataFrame()
            dt_df['Age'] = dt_res.age_cin
            dt_df['Health event'] = 'CIN'
            dt_df['location'] = location
            dt_dfs += dt_df

            dt_df = pd.DataFrame()
            dt_df['Age'] = dt_res.age_cancer
            dt_df['Health event'] = 'Cancer'
            dt_df['location'] = location
            dt_dfs += dt_df
            df = pd.concat(dt_dfs)
            dfs += df

            sc.saveobj(f'results/natural_history.obj', dfs)
    else:
        dfs = sc.loadobj(f'results/natural_history.obj')
    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    set_font(12)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    axes = axes.flatten()
    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            location = locations[pn]

            age_causal = dfs[pn]
            sns.violinplot(x="Health event",
                           y="Age",
                           data=age_causal, ax=ax)
            if location == 'drc':
                title_country = 'DRC'
            else:
                title_country = location.capitalize()
            ax.set_title(title_country)
    fig.suptitle('Natural history comparison')
    fig.tight_layout()
    pl.savefig(f"figures/natural_history.png", dpi=100)
    return

def plot_cancer_outcomes(locations, do_run=True):
    sims = []
    if do_run:
        for location in locations:
            if location in ['tanzania', 'myanmar']:
                calib_filestem = '_nov07'
            elif location in ['uganda', 'drc', 'uganda']:
                calib_filestem = '_nov06'
            else:
                calib_filestem = '_nov13'
            calib_pars = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')
            sim = rs.run_sim(location=location, calib_pars=calib_pars, end=2020, n_agents=50e3,
                             analyzers=[an.outcomes_by_year(start_year=2000),
                                        an.cum_dist(start_year=2000)])
            sim.save(f'results/{location}.sim')
            sims.append(sim)
    else:
        for location in locations:
            sim = hpv.load(f'results/{location}.sim')
            sims.append(sim)

    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=12)
    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    stacked_colors = [
        '#2db5a7',  # cleared
        '#eddc42',  # persisted
        '#e67f2c',  # progressed
        '#871a6c',  # cancer
    ]
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    axes = axes.flatten()
    ####################
    # Make plots
    ####################

    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            location = locations[pn]
            sim = sims[pn]

            a = sim.get_analyzer('outcomes_by_year')
            res = a.results
            years = a.durations

            df2 = pd.DataFrame()
            total_persisted = res["total"] - res["cleared"]
            df2["years"] = years
            df2["prob_persisted"] = (res["persisted"]) / total_persisted * 100
            df2["prob_progressed"] = (res["progressed"]) / total_persisted * 100
            df2["prob_cancer"] = (res["cancer"] + res["dead"]) / total_persisted * 100

            bottom = np.zeros(len(df2["years"]))
            layers = ["prob_persisted","prob_progressed", "prob_cancer"]
            labels = ["Persistent HPV", "CIN2+", "Cancer"]
            for ln, layer in enumerate(layers):
                ax.fill_between(
                    df2["years"],
                    bottom,
                    bottom + df2[layer],
                    color=stacked_colors[ln + 1],
                    label=labels[ln],
                )
                bottom += df2[layer]
            ax.set_xlabel("Time since infection")

            if pn == 0:
                ax.legend(loc='lower right')
            if location == 'drc':
                ax.set_title(location.upper())
            else:
                ax.set_title(location.capitalize())
    fig.tight_layout()
    pl.savefig(f"figures/cancer_outcomes.png", dpi=100)

    return

def plot_infection_outcomes(locations, do_run=True):
    sims = []
    if do_run:
        for location in locations:
            if location in ['tanzania', 'myanmar']:
                calib_filestem = '_nov07'
            elif location in ['uganda', 'drc', 'uganda']:
                calib_filestem = '_nov06'
            else:
                calib_filestem = '_nov13'
            calib_pars = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')
            sim = rs.run_sim(location=location, calib_pars=calib_pars, end=2020, n_agents=1e6, ms_agent_ratio=1,
                             analyzers=[an.outcomes_by_year(start_year=2000)])
            sim.save(f'results/{location}.sim')
            sims.append(sim)
    else:
        for location in locations:
            sim = hpv.load(f'results/{location}.sim')
            sims.append(sim)

    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=12)
    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    stacked_colors = [
        '#2db5a7',  # cleared
        '#eddc42',  # persisted
        '#e67f2c',  # progressed
        '#871a6c',  # cancer
    ]
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    axes = axes.flatten()
    ####################
    # Make plots
    ####################

    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            location = locations[pn]
            sim = sims[pn]

            a = sim.get_analyzer('outcomes_by_year')
            res = a.results
            years = a.durations

            df = pd.DataFrame()
            total_alive = res["total"]
            df["years"] = years
            df["prob_clearance"] = (res["cleared"]) / total_alive * 100
            df["prob_persist"] = (res["persisted"]) / total_alive * 100
            df["prob_progressed"] = (res["progressed"] + res["cancer"] + res["dead"]) / total_alive * 100

            end_ind = int(1 / (a.durations[1] - a.durations[0])) * 10
            bottom = np.zeros(len(df["years"][:end_ind]))
            layers = [
                "prob_clearance",
                "prob_persist",
                "prob_progressed",
            ]
            labels = ["Cleared", "Persistent infection", "CIN2+"]
            for ln, layer in enumerate(layers):
                ax.fill_between(
                    df["years"][:end_ind], bottom, bottom + df[layer][:end_ind], color=stacked_colors[ln],
                    label=labels[ln]
                )
                bottom += df[layer][:end_ind]

            ax.set_xlabel("Time since infection")

            if pn == 0:
                ax.legend(loc='lower right')
            if location == 'drc':
                ax.set_title(location.upper())
            else:
                ax.set_title(location.capitalize())
    fig.tight_layout()
    pl.savefig(f"figures/infection_outcomes.png", dpi=100)

    return


def plot_hpv_prevalence(locations, do_run=True):
    sims = []
    if do_run:
        for location in locations:
            if location in ['tanzania', 'myanmar']:
                calib_filestem = '_nov07'
            elif location in ['uganda', 'drc']:
                calib_filestem = '_nov06'
            else:
                calib_filestem = '_nov13'
            calib_pars = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')
            sim = rs.run_sim(location=location, calib_pars=calib_pars, end=2020, n_agents=50e3,
                             analyzers=[an.outcomes_by_year(start_year=2000),
                                        an.cum_dist(start_year=2000)])
            sim.save(f'results/{location}.sim')
            sims.append(sim)
    else:
        for location in locations:
            sim = hpv.load(f'results/{location}.sim')
            sims.append(sim)

    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=12)
    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    colors = sc.gridcolors(20)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10), sharey=True)
    axes = axes.flatten()
    ####################
    # Make plots
    ####################

    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            location = locations[pn]
            sim = sims[pn]
            prev = sim.results['precin_prevalence_by_age'][:,-1]
            cin_prev = sim.results['cin_prevalence_by_age'][:, -1]
            age = sim.pars['age_bin_edges'][:-1]
            ax.plot(age, 100*prev, label='HPV')
            ax.plot(age, 100 *cin_prev, label='CIN2+')
            if pn == 0:
                ax.legend()
            ax.set_xlabel('Age')
            sc.SIticks(ax)
            if location == 'drc':
                ax.set_title(location.upper())
            else:
                ax.set_title(location.capitalize())
    fig.tight_layout()
    pl.savefig(f"figures/hpv_prevalence.png", dpi=100)

    return


def plot_hpv_progression(locations, do_run=True):
    sims = []
    if do_run:
        for location in locations:
            if location in ['tanzania', 'myanmar']:
                calib_filestem = '_nov07'
            elif location in ['uganda', 'drc']:
                calib_filestem = '_nov06'
            else:
                calib_filestem = '_nov13'
            calib_pars = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')
            sim = rs.run_sim(location=location, calib_pars=calib_pars, end=2020, n_agents=50e3,
                             analyzers=[an.outcomes_by_year(start_year=2000),
                                        an.cum_dist(start_year=2000)])
            sim.save(f'results/{location}.sim')
            sims.append(sim)
    else:
        for location in locations:
            sim = hpv.load(f'results/{location}.sim')
            sims.append(sim)

    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=12)
    n_plots = len(locations)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    colors = sc.gridcolors(20)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    axes = axes.flatten()


    ####################
    # Make plots
    ####################

    for pn, ax in enumerate(axes):

        if pn >= len(locations):
            ax.set_visible(False)
        else:
            dt = 0.25
            this_precinx = np.arange(dt, 10 + dt, dt)
            location = locations[pn]
            sim = sims[pn]
            genotypes = ['hpv16', 'hpv18' , 'hi5', 'ohr']
            dur_cin = sc.autolist()
            cancer_fns = sc.autolist()
            cin_fns = sc.autolist()
            dur_precin = sc.autolist()
            for gi, genotype in enumerate(genotypes):
                dur_precin += sim['genotype_pars'][genotype]['dur_precin']
                dur_cin += sim['genotype_pars'][genotype]['dur_cin']
                cancer_fns += sim['genotype_pars'][genotype]['cancer_fn']
                cin_fns += sim['genotype_pars'][genotype]['cin_fn']

            # Panel A: clearance rates
            for gi, genotype in enumerate(genotypes):
                dysp = hppar.compute_severity(this_precinx[:], pars=cin_fns[gi])
                ax.plot(this_precinx, dysp, color=colors[gi], lw=2, label=genotype.upper())
            # ax.set_xticks(annual_x)
            ax.set_ylabel("Progression probabilities")
            ax.set_xlabel("Years since infection")
            ax.set_ylim([0, 1])
            if pn == 0:
                ax.legend()
            sc.SIticks(ax)
            if location == 'drc':
                ax.set_title(location.upper())
            else:
                ax.set_title(location.capitalize())
    fig.tight_layout()
    pl.savefig(f"figures/hpv_progression.png", dpi=100)

    return