"""
Shared utilities for comparing pulse (multi-age cohort) vaccination delivery against
annual single-cohort delivery, using cumulative HPV infections simulated in HPVsim.

Loads results from `hpv_scenarios/results/vaccination_2026Sep_cov{0pNN}/` folders, one
per simulated coverage value, each containing all campaign frequencies at that coverage.
"""

import os
import re
import glob
import numpy as np
import sciris as sc


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))
FOLDER_GLOB = "vaccination_2026Sep_cov*"
FILESTEM = "_nov06"

SUM_START_YEAR = 2026  # sum infections from this year onward
REFERENCE_FREQUENCY = "annual"

RISK_LEVELS = ["NAT", "HR"]

COUNTRIES = {
    # central_coverage is the actual simulated grid point at zero coverage gain, i.e. the
    # country's real current coverage (61.65% Zambia, 91.00% Sierra Leone) rounded to the
    # 2 decimal places used when the coverage folders were named.
    "zambia": {"central_coverage": 0.62},
    "sierra_leone": {"central_coverage": 0.91},
}

# Campaign frequency -> age range string embedded in filenames, and interval in years.
FREQUENCIES = {
    "annual":       {"ages": "9-10", "interval_years": 1},
    "biennial":     {"ages": "9-11", "interval_years": 2},
    "triennial":    {"ages": "9-12", "interval_years": 3},
    "quadrennial":  {"ages": "9-13", "interval_years": 4},
    "quinquennial": {"ages": "9-14", "interval_years": 5},
    "sextennial":   {"ages": "9-15", "interval_years": 6},
    "septennial":   {"ages": "9-16", "interval_years": 7},
    "octennial":    {"ages": "9-17", "interval_years": 8},
    "nonennial":    {"ages": "9-18", "interval_years": 9},
    "decennial":    {"ages": "9-19", "interval_years": 10},
}

SHORT_PULSE_SET = ["biennial", "triennial", "quadrennial", "quinquennial"]
FULL_PULSE_SET = SHORT_PULSE_SET + ["sextennial", "septennial", "octennial", "nonennial", "decennial"]

FREQUENCY_SETS = {
    "2to5": SHORT_PULSE_SET,
    "2to10": FULL_PULSE_SET,
}


def parse_coverage_from_folder(folder_name):
    """Extract coverage as a float from folder name like 'vaccination_2026Sep_cov0p62'."""
    m = re.search(r"cov(\d+)p(\d+)", folder_name)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def discover_coverage_folders(country):
    """Return list of (coverage, folder_path) for a country, sorted by coverage.

    Each coverage folder holds results for whichever country it was run for; folders are
    filtered to those containing this country's baseline file.
    """
    pattern = os.path.join(RESULTS_ROOT, FOLDER_GLOB)
    marker = f"{country}_baseline_NAT{FILESTEM}.obj"
    found = []
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        cov = parse_coverage_from_folder(os.path.basename(path))
        if cov is None:
            continue
        if not os.path.exists(os.path.join(path, marker)):
            continue
        found.append((cov, path))
    found.sort()
    return found


def scenario_filename(country, frequency, risk):
    """Build the .obj filename for a given country/frequency/risk combination."""
    ages = FREQUENCIES[frequency]["ages"]
    return f"{country}_vx_{frequency}_{ages}_{risk}{FILESTEM}.obj"


def load_msim(folder, country, frequency, risk):
    """Load a single msim .obj. Returns None if missing or on error."""
    filename = scenario_filename(country, frequency, risk)
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    try:
        return sc.loadobj(path)
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def cumulative_infections_per_seed(msim, start_year=SUM_START_YEAR):
    """Return a numpy array of cumulative infections (from start_year onward), one entry per seed."""
    sims = getattr(msim, "sims", None)
    if sims:
        totals = []
        for sim in sims:
            years = np.asarray(sim.results.year)
            infections = np.asarray(sim.results.infections)
            mask = years >= start_year
            totals.append(float(infections[mask].sum()))
        return np.asarray(totals)
    years = np.asarray(msim.results.year)
    infections = np.asarray(msim.results.infections)
    mask = years >= start_year
    return np.asarray([float(infections[mask].sum())])


def population_at_year(msim, year=SUM_START_YEAR):
    """Return mean (across seeds) n_alive at the given year, used as the per-10,000 denominator."""
    sims = getattr(msim, "sims", None) or [msim]
    vals = []
    for sim in sims:
        years = np.asarray(sim.results.year)
        idx = int(np.argmin(np.abs(years - year)))
        vals.append(float(sim.results.n_alive[idx]))
    return float(np.mean(vals))


def collect_frequency_data(country, risk, frequencies):
    """
    Build data[frequency] = list of (coverage, per_seed_infections_array), sorted by coverage,
    across all discovered coverage folders for this country. Also returns the population
    denominator (n_alive at SUM_START_YEAR), taken from the central-coverage annual scenario.
    """
    coverage_folders = discover_coverage_folders(country)
    data = {freq: [] for freq in frequencies}
    population = None
    central_coverage = COUNTRIES[country]["central_coverage"]

    for cov, folder in coverage_folders:
        for freq in frequencies:
            msim = load_msim(folder, country, freq, risk)
            if msim is None:
                continue
            per_seed = cumulative_infections_per_seed(msim)
            data[freq].append((cov, per_seed))
            if population is None and freq == REFERENCE_FREQUENCY and abs(cov - central_coverage) < 1e-6:
                population = population_at_year(msim)

    for freq in data:
        data[freq].sort(key=lambda t: t[0])

    return data, population


def reference_annual_value(data, central_coverage, population):
    """
    Return (mean, sem) cumulative infections per 10,000 population for the annual scenario
    at the country's central coverage. Requires 'annual' to be present in data (call
    collect_frequency_data with 'annual' included in the frequency list).
    """
    annual_series = dict(data["annual"])
    per_seed = annual_series.get(central_coverage)
    if per_seed is None:
        raise ValueError(f"No annual scenario found at central coverage {central_coverage}")
    per_10k = np.asarray(per_seed) / (population / 1e4)
    mean = float(np.mean(per_10k))
    sem = float(np.std(per_10k, ddof=1) / np.sqrt(len(per_10k))) if len(per_10k) > 1 else 0.0
    return mean, sem


def fit_breakeven(x, y, yerr=None):
    """
    Fit a straight line y = intercept + slope * x across all grid points (weighted by
    1/yerr if provided, i.e. points with smaller seed-to-seed uncertainty count more),
    then solve for the coverage gain x0 = -intercept / slope at which the fit crosses
    zero. The uncertainty on x0 is propagated from the covariance of the fit parameters
    via the delta method:

        Var(x0) = Var(intercept)/slope^2 + intercept^2 * Var(slope)/slope^4
                  - 2 * intercept * Cov(intercept, slope) / slope^3

    Returns (x0, x0_err), both None if the fit is degenerate (near-zero slope) or fewer
    than 3 points are available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None, None

    w = None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        if np.all(np.isfinite(yerr)) and np.all(yerr > 0):
            w = 1.0 / yerr

    try:
        (slope, intercept), cov = np.polyfit(x, y, 1, w=w, cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    if slope == 0 or not np.isfinite(slope):
        return None, None

    var_slope, var_intercept, cov_si = cov[0, 0], cov[1, 1], cov[0, 1]
    x0 = -intercept / slope
    var_x0 = (
        var_intercept / slope ** 2
        + intercept ** 2 * var_slope / slope ** 4
        - 2 * intercept * cov_si / slope ** 3
    )
    x0_err = float(np.sqrt(var_x0)) if var_x0 >= 0 and np.isfinite(var_x0) else None
    return float(x0), x0_err
