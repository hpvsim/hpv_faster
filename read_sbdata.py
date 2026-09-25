"""
Read in sexual behavior data
"""

# Import packages
import sciris as sc
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
from scipy.stats import norm, lognorm
import hpvsim as hpv
import hpvsim.utils as hpu

# Imports from this repository
import utils as ut
import run_sim as rs
import locations as set

def percentiles_to_pars(x1, p1, x2, p2):
    """ Find the parameters of a normal distribution where:
            P(X < p1) = x1
            P(X < p2) = x2
    """
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)

    location = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = (x2 - x1) / (p2ppf - p1ppf)
    return location, scale


def logn_percentiles_to_pars(x1, p1, x2, p2):
    """ Find the parameters of a lognormal distribution where:
            P(X < p1) = x1
            P(X < p2) = x2
    """
    x1 = np.log(x1)
    x2 = np.log(x2)
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)
    s = (x2 - x1) / (p2ppf - p1ppf)
    mean = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = np.exp(mean)
    return s, scale


def read_debut_data(dist_type='lognormal'):
    '''
    Read in dataframes taken from DHS and return them in a plot-friendly format,
    optionally saving the distribution parameters. 
    Uses interpolation to find 25th and 50th percentiles from the data.
    '''

    df1 = pd.read_csv('data/afs_dist.csv')
    
    # Create a median dataframe for backwards compatibility with plotting functions
    df2 = pd.DataFrame()  # Will be populated with interpolated medians

    # Rearrange data into a plot-friendly format
    dff = {}
    rvs = {'Women': {}, 'Men': {}}
    
    # Store interpolated medians for plotting compatibility
    median_data = {'Country': [], 'Women median': [], 'Men median': []}

    for sex in ['Women', 'Men']:

        # Dynamically extract age columns from the CSV
        sex_columns = [col for col in df1.columns if col.startswith(f'{sex} ')]
        age_columns = ['Country'] + sex_columns
        dfw = df1[age_columns]
        dfw = dfw.melt(id_vars='Country', value_name='Percentage', var_name='AgeStr')
        
        print(f"Processing {sex} data:")
        print(f"Found columns: {sex_columns}")
        print(f"Sample data for first country:")
        print(dfw[dfw['Country'] == dfw['Country'].iloc[0]].head())

        # Add values for proportion ever having sex (only if 'never' column exists)
        countries = dfw.Country.unique()
        n_countries = len(countries)
        has_never_column = any('never' in col for col in sex_columns)
        
        if has_never_column:
            vals = []
            for country in countries:
                val = 100-dfw.loc[(dfw['AgeStr'] == f'{sex} never') & (dfw['Country'] == country) , 'Percentage'].iloc[0]
                vals.append(val)

            data_cat = {'Country': countries, 'AgeStr': [f'{sex} 60']*n_countries}
            data_cat["Percentage"] = vals
            df_cat = pd.DataFrame.from_dict(data_cat)
            dfw = pd.concat([dfw,df_cat])

        # Dynamically extract age values from column names and create conditions
        age_values = []
        conditions = []
        for col in sex_columns:
            if 'never' in col:
                continue  # Skip 'never' column for age mapping
            # Extract numeric age from column name (e.g., "Women 15" -> 15)
            age_str = col.split(' ', 1)[1]  # Get everything after first space
            try:
                age = int(age_str)
                age_values.append(age)
                conditions.append((dfw['AgeStr'] == col))
            except ValueError:
                # Handle non-numeric age strings like "never" - skip them
                pass
        
        # Add the artificial "60" age for "ever had sex" proportion (only if 'never' column existed)
        if has_never_column:
            age_values.append(60)
            conditions.append((dfw['AgeStr'] == f"{sex} 60"))
        
        dfw['Age'] = np.select(conditions, age_values)

        dff[sex] = dfw

        res = dict()
        res["location"] = []
        res["par1"] = []
        res["par2"] = []
        res["dist"] = []
        for pn,country in enumerate(countries):
            # Filter out 'never' and artificial '60' columns if they exist
            exclude_conditions = (dfw["Country"] == country)
            if has_never_column:
                exclude_conditions = exclude_conditions & (dfw["AgeStr"] != f'{sex} never') & (dfw["AgeStr"] != f'{sex} 60')
            else:
                exclude_conditions = exclude_conditions & True  # No additional filtering needed
            dfplot = dfw.loc[exclude_conditions]
            
            if len(dfplot) == 0:
                continue  # Skip if no data available
                
            # Get ages and percentages for interpolation
            if has_never_column:
                available_ages = sorted([age for age in age_values if age != 60])  # Exclude artificial 60
            else:
                available_ages = sorted(age_values)  # Use all available ages
            if not available_ages:
                continue
            
            # Create arrays for interpolation
            ages = []
            percentages = []
            for age in available_ages:
                pct_data = dfplot.loc[dfplot["Age"] == age, 'Percentage']
                if len(pct_data) > 0:
                    ages.append(age)
                    percentages.append(pct_data.iloc[0])
            
            if len(ages) < 2:
                continue  # Need at least 2 points for interpolation
                
            ages = np.array(ages)
            percentages = np.array(percentages)
            # Interpolate to find 10th percentile age
            if percentages.max() >= 10:
                x1 = np.interp(10, percentages, ages)
                p1 = 0.10
            else:
                # If we don't reach 10%, use the highest percentage available
                max_idx = np.argmax(percentages)
                x1 = ages[max_idx]
                p1 = percentages[max_idx] / 100
                
            # Interpolate to find 50th percentile (median) age
            if percentages.max() >= 50:
                x2 = np.interp(50, percentages, ages)
                p2 = 0.50
            else:
                # If we don't reach 50%, use a higher percentile if available
                if percentages.max() >= 40:
                    x2 = np.interp(percentages.max(), percentages, ages)
                    p2 = percentages.max() / 100
                else:
                    continue  # Skip if insufficient data range
                    
            # Store interpolated median for plotting compatibility
            if sex == 'Women':
                if country not in [c for c in median_data['Country']]:
                    median_data['Country'].append(country)
                    median_data['Women median'].append(x2)
                    median_data['Men median'].append(None)  # Will be filled when processing men
                else:
                    # Update the existing entry
                    idx = median_data['Country'].index(country)
                    median_data['Women median'][idx] = x2
            else:  # Men
                if country not in [c for c in median_data['Country']]:
                    median_data['Country'].append(country)
                    median_data['Women median'].append(None)
                    median_data['Men median'].append(x2)
                else:
                    # Update the existing entry
                    idx = median_data['Country'].index(country)
                    median_data['Men median'][idx] = x2
            
            res["location"].append(country)
            res["dist"].append(dist_type)

            if dist_type=='normal':
                loc, scale = percentiles_to_pars(x1, p1, x2, p2)
                rv = norm(loc=loc, scale=scale)
                res["par1"].append(loc)
                res["par2"].append(scale)
            elif dist_type=='lognormal':
                s, scale = logn_percentiles_to_pars(x1, p1, x2, p2)
                rv = lognorm(s=s, scale=scale)
                res["par1"].append(rv.mean())
                res["par2"].append(rv.std())

            rvs[sex][country] = rv

        pd.DataFrame.from_dict(res).to_csv(f'data/sb_pars_{sex.lower()}_{dist_type}.csv')

    # Create df2 for backwards compatibility with plotting functions
    df2 = pd.DataFrame(median_data)
    df2['y'] = 50  # For plotting compatibility

    return countries, dff, df2, rvs


def read_marriage_data():
    dfraw = pd.read_csv('data/prop_married.csv')
    df = dfraw.melt(id_vars=['Country', 'Survey'], value_name='Percentage', var_name='AgeRange')
    return df


def get_sb_from_sims(dist_type='lognormal', marriage_scale=1, debut_bias=[0,0],
                     verbose=-1, debug=False):
    '''
    Run sims with the sexual debut parameters inferred from DHA data, and save
    the proportion of people of each age who've ever had sex
    '''

    locations = set.locations
    countries_to_run = locations
    sims = rs.run_sims(
        locations=countries_to_run,
        age_pyr=True,
        analyzers=[ut.AFS(),ut.prop_married(),hpv.snapshot(timepoints=['2020'])],
        debug=debug,
        dist_type=dist_type,
        marriage_scale=marriage_scale,
        debut_bias=debut_bias,
        verbose=verbose,
    )

    # Save output on age at first sex (AFS)
    dfs = sc.autolist()
    for country in countries_to_run:
        a = sims[country].get_analyzer('AFS')
        for cs,cohort_start in enumerate(a.cohort_starts):
            df = pd.DataFrame()
            df['age'] = a.bins
            df['cohort'] = cohort_start
            df['model_prop_f'] = a.prop_active_f[cs,:]
            df['model_prop_m'] = a.prop_active_m[cs,:]
            df['country'] = country
            dfs += df
    afs_df = pd.concat(dfs)
    sc.saveobj(f'results/model_sb_AFS.obj', afs_df)

    # Save output on proportion married
    dfs = sc.autolist()
    for country in countries_to_run:
        a = sims[country].get_analyzer('prop_married')
        df = a.df
        df['country'] = country
        dfs += df
    pm_df = pd.concat(dfs)
    sc.saveobj(f'results/model_sb_prop_married.obj', pm_df)

    # Save output on age differences between partners
    dfs = sc.autolist()
    for country in countries_to_run:
        df = pd.DataFrame()
        snapshot = sims[country].get_analyzer('snapshot')
        ppl = snapshot.snapshots[0]
        age_diffs = ppl.contacts['m']['age_m'] - ppl.contacts['m']['age_f']
        df['age_diffs'] = age_diffs
        df['country'] = country
        dfs += df
    agediff_df = pd.concat(dfs)
    sc.saveobj(f'results/model_age_diffs.obj', agediff_df)


    # Save output on the number of casual relationships
    binspan = 5
    bins = np.arange(15, 50, binspan)
    dfs = sc.autolist()
    for country in countries_to_run:
        snapshot = sims[country].get_analyzer('snapshot')
        ppl = snapshot.snapshots[0]
        conditions = {}
        general_conditions = ppl.is_female * ppl.alive * ppl.level0 * ppl.is_active
        for ab in bins:
            conditions[ab] = (ppl.age >= ab) * (ppl.age < ab + binspan) * general_conditions

        casual_partners = {(0,1): sc.autolist(), (1,2):sc.autolist(), (2,3):sc.autolist(), (3,5):sc.autolist(), (5,50):sc.autolist()}
        for cp in casual_partners.keys():
            for ab,age_cond in conditions.items():
                this_condition = conditions[ab] * (ppl.current_partners[1,:]>=cp[0]) * (ppl.current_partners[1,:]<cp[1])
                casual_partners[cp] += len(hpu.true(this_condition))

        popsize = sc.autolist()
        for ab, age_cond in conditions.items():
            popsize += len(hpu.true(age_cond))

        # Construct dataframe
        n_bins = len(bins)
        partners = np.repeat([0, 1, 2, 3, 5], n_bins)
        allbins = np.tile(bins, 5)
        counts = np.concatenate([val for val in casual_partners.values()])
        allpopsize = np.tile(popsize, 5)
        shares = counts / allpopsize
        datadict = dict(bins=allbins, partners=partners, counts=counts, popsize=allpopsize, shares=shares)
        df = pd.DataFrame.from_dict(datadict)
        df['country'] = country
        dfs += df

    casual_df = pd.concat(dfs)
    sc.saveobj(f'results/model_casual.obj', casual_df)


    return sims, afs_df, pm_df, agediff_df, casual_df


def plot_sb(dist_type='lognormal'):
    '''
    Create plots of sexual behavior inputs and outputs
    '''

    ut.set_font(12)

    data_countries, dff, df2, rvs = read_debut_data(dist_type=dist_type)
    alldf = sc.loadobj(f'results/model_sb_AFS.obj')
    countries = alldf['country'].unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)

    for sk,sex in {'f':'Women', 'm':'Men'}.items():
        fig, axes = pl.subplots(n_rows, n_cols, figsize=(8,11))
        axes = axes.flatten()
        dfw = dff[sex]

        for pn,country in enumerate(countries):
            ax = axes[pn]
            data_country = ut.map_sb_loc(country)
            dfplot = dfw.loc[(dfw["Country"]==data_country)&(dfw["AgeStr"]!=f'{sex} never')&(dfw["AgeStr"]!=f'{sex} 60')]
            dfmed = df2.loc[df2["Country"] == data_country]
            if len(dfplot)>0:
                sns.scatterplot(ax=ax, data=dfplot, x="Age", y="Percentage")
                sns.scatterplot(ax=ax, data=dfmed, x=f"{sex} median", y="y")

                rv = rvs[sex][data_country]
                xx = np.arange(12,30.1,0.1)
                xxx = np.arange(12,31,1)
                ax.plot(xx, rv.cdf(xx)*100, 'k--', lw=2)

            for cohort in alldf["cohort"].unique():
                modely = np.array(alldf.loc[(alldf["country"]==country)&(alldf["cohort"]==cohort)][f'model_prop_{sk}'])
                ax.plot(xxx, modely*100, 'b-', lw=1)
            title_country = country.title()
            if title_country == 'Congo Democratic Republic':
                title_country = 'DRC'

            ax.set_title(title_country)
            ax.set_ylabel('')
            ax.set_xlabel('')

        fig.tight_layout()
        pl.savefig(f"figures/SMs/fig_sb_{sex.lower()}.png", dpi=100)

    return


def plot_prop_married():
    '''
    Create plots of sexual behavior inputs and outputs
    '''

    ut.set_font(12)
    # Read in data and model results
    df = read_marriage_data()
    modeldf = sc.loadobj(f'results/model_sb_prop_married.obj')
    modeldf.reset_index()

    # Plot
    countries = modeldf.country.unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)
    colors = sc.gridcolors(1)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(8, 11))
    if n_countries>1:
        axes = axes.flatten()
    else:
        axes = axes

    for pn, country in enumerate(countries):
        if n_countries > 1:
            ax = axes[pn]
        else:
            ax = axes

        # Plot data
        d_country = ut.map_sb_loc(country)
        dfplot_d = df.loc[(df["Country"] == d_country)]
        sns.scatterplot(ax=ax, data=dfplot_d, x="AgeRange", y="Percentage")

        # Plot model
        location = ut.rev_map_sb_loc(country)
        dfplot_m = modeldf.loc[modeldf['country']==location]
        # dfplot_m['val'] = dfplot_m['val'].multiply(100)
        dfplot_m['val'] = dfplot_m['val'].apply(lambda x: x * 100)
        sns.boxplot(data=dfplot_m, x="age", y="val", color=colors[0], ax=ax)

        title_country = country.title()
        if title_country == 'Drc':
            title_country = 'DRC'

        ax.set_title(title_country)
        ax.set_ylabel('')
        ax.set_xlabel('')

    fig.tight_layout()
    pl.savefig(f"figures/SMs/fig_prop_married.png", dpi=100)

    return



def plot_age_diffs():
    '''
    Plot the age differences between marital partners
    '''

    ut.set_font(12)
    agediffs = sc.loadobj(f'results/model_age_diffs.obj')

    # Plot
    countries = agediffs.country.unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)
    colors = sc.gridcolors(1)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(8, 11))
    if n_countries>1:
        axes = axes.flatten()
    else:
        axes = axes

    for pn, country in enumerate(countries):
        if n_countries > 1:
            ax = axes[pn]
        else:
            ax = axes

        # Plot model
        dfplot_m = agediffs.loc[agediffs['country']==country]
        sns.kdeplot(data=dfplot_m, color=colors[0], ax=ax)
        ax.legend([], [], frameon=False)

        title_country = country.title()
        if title_country == 'Drc':
            title_country = 'DRC'

        ax.set_title(title_country)
        ax.set_ylabel('')
        ax.set_xlabel('')

    fig.tight_layout()
    pl.savefig(f"figures/SMs/fig_age_diffs.png", dpi=100)

    return


def plot_casuals():
    '''
    Plot the number of casual partners by age
    '''

    ut.set_font(12)
    casual_df = sc.loadobj(f'results/model_casual.obj')
    data_df = pd.read_excel(f'data/casuals.xlsx')
    data_df = data_df.melt(id_vars=['Country', 'Survey'], value_name='Percentage', var_name='AgeStr')

    # Plot
    countries = casual_df.country.unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(8, 11))
    if n_countries>1:
        axes = axes.flatten()
    else:
        axes = axes

    for pn, country in enumerate(countries):
        if n_countries > 1:
            ax = axes[pn]
        else:
            ax = axes

        # Plot model
        dfplot_m = casual_df.loc[(casual_df['country']==country) & (casual_df['partners'] > 0)]
        dfplot_m['shares'] = dfplot_m['shares'].apply(lambda x: x * 100)
        dfpm = pd.DataFrame({'bins':dfplot_m.bins.unique(), 'shares':dfplot_m.groupby(['bins', 'country']).sum()['shares'].values})
        sns.barplot(data=dfpm, x="bins", y="shares", ax=ax, color='b', alpha=0.5)  #hue="partners", ax=ax)
        ax.legend([], [], frameon=False)

        # Plot data
        data_country = ut.map_sb_loc(country)
        dfplot_d = data_df.loc[data_df['Country']==data_country]
        sns.scatterplot(data=dfplot_d, x="AgeStr", y="Percentage", ax=ax, marker="D", color="k", s=50)

        title_country = country.title()
        if title_country == 'Drc':
            title_country = 'DRC'

        ax.set_title(title_country)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    pl.savefig(f"figures/SMs/fig_casual.png", dpi=100)

    return


#%% Run as a script
if __name__ == '__main__':

    dist_type = 'lognormal'
    countries, dff, df2, rvs = read_debut_data(dist_type=dist_type)
    do_run = False

    if do_run:
        sims, afs_df, pm_df, agediff_df, casual_df = get_sb_from_sims(
            dist_type=dist_type,
            marriage_scale=1,
            debut_bias=[-1,-1],
            debug=False,
            verbose=0.1
        )

    # Plotting functions
    # plot_sb(dist_type=dist_type)
    # plot_prop_married()
    # plot_age_diffs()
    # plot_casuals()

    print('Done.')
