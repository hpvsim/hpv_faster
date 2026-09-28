"""
Breakeven summary plots: for each country and risk level, the coverage gain a pulse
(multi-age cohort) vaccination strategy needs over the current annual single-cohort
coverage to match it in cumulative HPV infections (2026 onward), as a function of the
pulse interval.

Reuses the per-frequency shortfall curves from plot_pulse_vs_annual.py (mean cumulative
infections difference vs. coverage gain across all 15 simulated coverage grid points) and
fits each to a weighted straight line (weights = 1/SEM at each grid point) to find its
zero-crossing, with the crossing's uncertainty propagated from the fit parameters'
covariance (see pulse_infections_common.fit_breakeven).

Output:
  hpv_scenarios/analysis/figures/breakeven_summary/{country}_{risk}_{variant}.png
  hpv_scenarios/analysis/figures/breakeven_summary/breakeven_coverage_gains.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import pulse_infections_common as common
from plot_pulse_vs_annual import build_series


def compute_breakevens(country, risk, frequencies):
    """Returns dict[freq] = (coverage_gain_pp, coverage_gain_pp_err), both None if no fit."""
    series = build_series(country, risk, frequencies)
    result = {}
    for freq in frequencies:
        if freq not in series:
            result[freq] = (None, None)
            continue
        x, y, yerr = series[freq]
        result[freq] = common.fit_breakeven(x, y, yerr)
    return result


def plot_variant(country, risk, variant, frequencies, breakevens, output_path):
    intervals = [common.FREQUENCIES[f]["interval_years"] for f in frequencies]
    gains = [breakevens[f][0] for f in frequencies]
    gain_errs = [breakevens[f][1] for f in frequencies]

    fig, ax = plt.subplots(figsize=(7, 5))
    found = [(i, g, (e if e is not None else 0.0))
             for i, g, e in zip(intervals, gains, gain_errs) if g is not None]
    missing_x = [i for i, g in zip(intervals, gains) if g is None]

    if found:
        found_x, found_y, found_yerr = zip(*found)
        ax.errorbar(found_x, found_y, yerr=found_yerr, marker="o", color="tab:blue",
                    linewidth=1.5, capsize=4)
    if missing_x:
        ax.scatter(missing_x, [0] * len(missing_x), marker="x", color="tab:red",
                   label="No break-even from fit", zorder=5)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Pulse interval (years)")
    ax.set_ylabel("Coverage gain required to break even\nwith annual delivery (percentage points)")
    ax.set_title(f"{country.replace('_', ' ').title()} {risk}: break-even coverage gain vs pulse interval")
    ax.set_xticks(intervals)
    if missing_x:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    output_dir = os.path.join(common.SCRIPT_DIR, "figures", "breakeven_summary")
    os.makedirs(output_dir, exist_ok=True)

    csv_rows = []
    for country in common.COUNTRIES:
        for risk in common.RISK_LEVELS:
            for variant, frequencies in common.FREQUENCY_SETS.items():
                print(f"{country} / {risk} / {variant}")
                breakevens = compute_breakevens(country, risk, frequencies)
                out_path = os.path.join(output_dir, f"{country}_{risk}_{variant}.png")
                plot_variant(country, risk, variant, frequencies, breakevens, out_path)
                if variant == "2to10":  # avoid duplicating the 2-5yr subset in the CSV
                    for freq in frequencies:
                        gain, gain_err = breakevens[freq]
                        csv_rows.append({
                            "country": country,
                            "risk": risk,
                            "frequency": freq,
                            "interval_years": common.FREQUENCIES[freq]["interval_years"],
                            "coverage_gain_pp": gain,
                            "coverage_gain_pp_err": gain_err,
                        })

    csv_path = os.path.join(output_dir, "breakeven_coverage_gains.csv")
    fieldnames = ["country", "risk", "frequency", "interval_years", "coverage_gain_pp", "coverage_gain_pp_err"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()
