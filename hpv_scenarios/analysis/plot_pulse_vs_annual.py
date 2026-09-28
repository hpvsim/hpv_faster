"""
Figure-3-style plots: cumulative HPV infections under pulse (multi-age cohort) vaccination
compared with annual single-cohort delivery, simulated in HPVsim.

For each country and risk level (NAT / HR / UHR), produces a figure where:
  - x-axis: pulse coverage relative to the country's current annual coverage (percentage points)
  - y-axis: difference in cumulative HPV infections per 10,000 population, 2026 onward
            (pulse minus annual delivery at its current coverage)
  - one line per pulse frequency, mean +/- combined SEM (pulse and reference) across seeds
  - a marker at each line's interpolated break-even point (where the difference crosses zero)

Two variants are produced per country/risk: '2to5' (biennial-quinquennial, matching the
manuscript's Figure 3) and '2to10' (biennial-decennial, using the full simulated range).

Output: hpv_scenarios/analysis/figures/pulse_vs_annual/{country}_{risk}_{variant}.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import pulse_infections_common as common


def build_series(country, risk, frequencies):
    """
    Returns dict[freq] = (x, y, yerr) where x is coverage gain (pp), y is the mean shortfall
    (pulse minus annual-at-central, per 10,000 population), yerr is the combined SEM.
    """
    all_freqs = list(dict.fromkeys(frequencies + [common.REFERENCE_FREQUENCY]))
    data, population = common.collect_frequency_data(country, risk, all_freqs)
    if population is None:
        raise RuntimeError(f"Could not determine population denominator for {country}/{risk}")

    central_coverage = common.COUNTRIES[country]["central_coverage"]
    ref_mean, ref_sem = common.reference_annual_value(data, central_coverage, population)

    series = {}
    for freq in frequencies:
        points = data.get(freq, [])
        if not points:
            continue
        covs, means, sems = [], [], []
        for cov, per_seed in points:
            per_10k = np.asarray(per_seed) / (population / 1e4)
            covs.append(cov)
            means.append(float(np.mean(per_10k)))
            sems.append(float(np.std(per_10k, ddof=1) / np.sqrt(len(per_10k))) if len(per_10k) > 1 else 0.0)
        covs = np.asarray(covs)
        means = np.asarray(means)
        sems = np.asarray(sems)

        x = (covs - central_coverage) * 100.0
        y = means - ref_mean
        yerr = np.sqrt(sems ** 2 + ref_sem ** 2)
        series[freq] = (x, y, yerr)

    return series


def plot_variant(country, risk, variant, frequencies, output_path):
    series = build_series(country, risk, frequencies)
    if not series:
        print(f"[{country}/{risk}/{variant}] No data found; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(frequencies)))

    for color, freq in zip(colors, frequencies):
        if freq not in series:
            continue
        x, y, yerr = series[freq]
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", color=color, label=freq.capitalize(),
            capsize=3, linewidth=1.5, linestyle="none",
        )
        x0, x0_err = common.fit_breakeven(x, y, yerr)
        if x0 is not None:
            x_fit = np.linspace(min(x.min(), x0), max(x.max(), x0), 100)
            slope, intercept = np.polyfit(x, y, 1, w=(1.0 / yerr if np.all(yerr > 0) else None))
            ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=1.0, alpha=0.6, zorder=1)
            xerr = x0_err if x0_err is not None else 0.0
            ax.errorbar(
                [x0], [0], xerr=[xerr],
                marker="*", color=color, markersize=16,
                markeredgecolor="black", markeredgewidth=0.5,
                capsize=4, zorder=5,
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"Pulse coverage vs current annual coverage (percentage points)\n"
                  f"(annual = {common.COUNTRIES[country]['central_coverage']:.1%})")
    ax.set_ylabel(f"Cumulative HPV infections per 10,000 population, {common.SUM_START_YEAR} onward\n"
                  f"(pulse minus annual, at annual's current coverage)")
    ax.set_title(f"{country.replace('_', ' ').title()} {risk}: pulse vs annual single-cohort delivery")
    ax.legend(title="Pulse frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    output_dir = os.path.join(common.SCRIPT_DIR, "figures", "pulse_vs_annual")
    os.makedirs(output_dir, exist_ok=True)

    for country in common.COUNTRIES:
        for risk in common.RISK_LEVELS:
            for variant, frequencies in common.FREQUENCY_SETS.items():
                print(f"{country} / {risk} / {variant}")
                out_path = os.path.join(output_dir, f"{country}_{risk}_{variant}.png")
                plot_variant(country, risk, variant, frequencies, out_path)


if __name__ == "__main__":
    main()
