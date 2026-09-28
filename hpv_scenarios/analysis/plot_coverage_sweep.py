"""
Plot HPV infections vs vaccination coverage for a country's risk scenarios.

For each risk scenario (HR, NAT), produces a figure where:
  - x-axis: coverage relative to reference (percentage points)
  - y-axis: cumulative HPV infections relative to the annual scenario at reference coverage
  - one line per non-annual campaign frequency (biennial, triennial, quadrennial, quinquennial)

Reads msim results from `hpv_scenarios/results/vaccination_2026Jun_cov{0pNN}/` folders.
Pass --country to specify the country (default: zambia).
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))
FOLDER_GLOB = "vaccination_2026Jun_cov*"
FILESTEM = "_nov06"

COUNTRY = "zambia"
RISK_SCENARIOS = ["NAT", "HR"]
REFERENCE_COVERAGE = 0.55
REFERENCE_CAMPAIGN = "annual"
SUM_START_YEAR = 2025  # sum infections from this year onward

CAMPAIGNS = {
    "annual":       "9-10",
    "biennial":     "9-11",
    "triennial":    "9-12",
    "quadrennial":  "9-13",
    "quinquennial": "9-14",
}
LINE_CAMPAIGNS = ["biennial", "triennial", "quadrennial", "quinquennial"]


def parse_coverage_from_folder(folder_name):
    """Extract coverage as a float from folder name like 'vaccination_2026Jun_cov0p55'."""
    m = re.search(r"cov(\d+)p(\d+)", folder_name)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def discover_coverage_folders(results_root):
    """Return list of (coverage, folder_path) sorted by coverage."""
    pattern = os.path.join(results_root, FOLDER_GLOB)
    found = []
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        cov = parse_coverage_from_folder(os.path.basename(path))
        if cov is None:
            continue
        found.append((cov, path))
    found.sort()
    return found


def cumulative_infections_per_sim(msim, start_year=SUM_START_YEAR):
    """
    Return a numpy array of cumulative infections (from start_year onward), one entry per
    underlying sim in the multisim. Falls back to the reduced mean if individual sims are
    not available.
    """
    sims = getattr(msim, "sims", None)
    if sims:
        totals = []
        for sim in sims:
            years = np.asarray(sim.results.year)
            infections = np.asarray(sim.results.infections)
            mask = years >= start_year
            totals.append(float(infections[mask].sum()))
        return np.asarray(totals)
    # Fallback: only the reduced mean is available
    years = np.asarray(msim.results.year)
    infections = np.asarray(msim.results.infections)
    mask = years >= start_year
    return np.asarray([float(infections[mask].sum())])


def load_infections(folder, campaign, risk):
    """Load a single msim and return per-seed cumulative infections. Returns None if missing."""
    ages = CAMPAIGNS[campaign]
    filename = f"{COUNTRY}_vx_{campaign}_{ages}_{risk}{FILESTEM}.obj"
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    try:
        msim = sc.loadobj(path)
        return cumulative_infections_per_sim(msim)
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def collect_data(coverage_folders):
    """
    Build a nested dict: data[risk][campaign] = list of (coverage, infections_per_seed)
    where infections_per_seed is a numpy array of length n_seeds. Sorted by coverage.
    """
    data = {risk: {camp: [] for camp in CAMPAIGNS} for risk in RISK_SCENARIOS}
    for cov, folder in coverage_folders:
        print(f"Coverage {cov:.2f} -> {os.path.basename(folder)}")
        for risk in RISK_SCENARIOS:
            for camp in CAMPAIGNS:
                per_seed = load_infections(folder, camp, risk)
                if per_seed is not None:
                    data[risk][camp].append((cov, per_seed))
    return data


def plot_one_risk(risk, data_for_risk, output_path):
    """Generate the coverage-sweep figure for one risk scenario with std error bars."""
    annual_series = dict(data_for_risk["annual"])
    ref_per_seed = annual_series.get(REFERENCE_COVERAGE)
    if ref_per_seed is None or np.mean(ref_per_seed) == 0:
        print(f"[{risk}] No annual @ {REFERENCE_COVERAGE:.0%} reference value; skipping plot.")
        return
    reference = float(np.mean(ref_per_seed))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(LINE_CAMPAIGNS)))

    for color, camp in zip(colors, LINE_CAMPAIGNS):
        series = sorted(data_for_risk[camp])
        if not series:
            continue
        covs = np.array([c for c, _ in series])
        means = np.array([np.mean(v) for _, v in series])
        # Sample standard error of the mean across seeds (ddof=1)
        sems = np.array([
            (np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            for _, v in series
        ])
        x = (covs - REFERENCE_COVERAGE) * 100.0  # percentage points
        y = means / reference
        yerr = sems / reference
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", color=color, label=camp.capitalize(),
            capsize=3, linewidth=1.5,
        )

    n_seeds_label = len(ref_per_seed)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Coverage change vs 55% (percentage points)")
    ax.set_ylabel(f"Cumulative HPV infections\n(relative to annual @ {REFERENCE_COVERAGE:.0%})")
    ax.set_title(f"{COUNTRY.replace('_', ' ').title()} {risk}: coverage vs HPV infections (error bars: SEM across {n_seeds_label} seeds)")
    ax.set_xlim(-11, 11)
    ax.legend(title="Campaign frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_one_risk_paired(risk, data_for_risk, output_path):
    """
    Like plot_one_risk, but compares sim-by-sim against the annual @ 55% scenario.
    Assumes seed i at any (coverage, campaign) shares its random seed with seed i at the
    reference, so dividing per-seed cancels shared stochasticity.
    """
    annual_series = dict(data_for_risk["annual"])
    ref_per_seed = annual_series.get(REFERENCE_COVERAGE)
    if ref_per_seed is None or np.any(np.asarray(ref_per_seed) == 0):
        print(f"[{risk}] No usable annual @ {REFERENCE_COVERAGE:.0%} reference; skipping paired plot.")
        return
    ref_per_seed = np.asarray(ref_per_seed, dtype=float)
    n_seeds_ref = len(ref_per_seed)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(LINE_CAMPAIGNS)))

    for color, camp in zip(colors, LINE_CAMPAIGNS):
        series = sorted(data_for_risk[camp])
        if not series:
            continue
        covs = []
        means = []
        sems = []
        for cov, per_seed in series:
            per_seed = np.asarray(per_seed, dtype=float)
            n = min(len(per_seed), n_seeds_ref)
            if n == 0:
                continue
            ratios = per_seed[:n] / ref_per_seed[:n]
            covs.append(cov)
            means.append(float(np.mean(ratios)))
            sems.append(float(np.std(ratios, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        if not covs:
            continue
        x = (np.asarray(covs) - REFERENCE_COVERAGE) * 100.0
        ax.errorbar(
            x, means, yerr=sems,
            marker="o", color=color, label=camp.capitalize(),
            capsize=3, linewidth=1.5,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Coverage change vs 55% (percentage points)")
    ax.set_ylabel(f"Cumulative HPV infections (per-seed ratio)\nvs annual @ {REFERENCE_COVERAGE:.0%}")
    ax.set_title(
        f"{COUNTRY.replace('_', ' ').title()} {risk}: coverage vs HPV infections "
        f"(paired by seed; error bars: SEM across {n_seeds_ref} seeds)"
    )
    ax.set_xlim(-11, 11)
    ax.legend(title="Campaign frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    global COUNTRY
    parser = argparse.ArgumentParser(description="Plot HPV infections vs vaccination coverage.")
    parser.add_argument("--country", default=COUNTRY, help="Country name (underscore-separated, e.g. sierra_leone)")
    args = parser.parse_args()
    COUNTRY = args.country

    coverage_folders = discover_coverage_folders(RESULTS_ROOT)
    if not coverage_folders:
        print(f"No coverage folders found under {RESULTS_ROOT}")
        return

    print(f"Found {len(coverage_folders)} coverage folders.")
    data = collect_data(coverage_folders)

    output_dir = os.path.join(SCRIPT_DIR, "figures", "coverage_sweep")
    os.makedirs(output_dir, exist_ok=True)

    for risk in RISK_SCENARIOS:
        out_path = os.path.join(output_dir, f"{COUNTRY}_{risk}_coverage_sweep.png")
        plot_one_risk(risk, data[risk], out_path)
        paired_path = os.path.join(output_dir, f"{COUNTRY}_{risk}_coverage_sweep_paired.png")
        plot_one_risk_paired(risk, data[risk], paired_path)


if __name__ == "__main__":
    main()
