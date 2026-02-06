#!/usr/bin/env python3
"""Compute statistical tests for selection strategy comparison.

Outputs:
- Cohen's h for 80% vs 50% binary comparison
- Paired Wilcoxon signed-rank test on log(speedup ratios)
- Exact sign test on binary fast_1 outcomes
- Median log(speedup ratio) as effect size
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def main():
    # Load selection results
    with open(Path("runs/analysis/selection_results_subset1.json")) as f:
        data = json.load(f)

    per_sweep = data["per_sweep"]

    # --- Cohen's h ---
    h = cohens_h(0.8, 0.5)
    print(f"Cohen's h (80% vs 50%): {h:.3f}")
    print(f"  Interpretation: {'large' if abs(h) >= 0.8 else 'medium-to-large' if abs(h) >= 0.5 else 'small-to-medium'} effect")
    print()

    # --- Paired speedup ratios ---
    tasks = ["4", "5", "12", "14", "15"]
    seeds = ["42", "43"]

    log_ratios = []
    surprisal_wins = 0
    confidence_wins = 0
    ties = 0

    print("Task-seed speedup pairs (surprisal vs confidence):")
    print(f"{'Seed':>6} {'Task':>6} {'Surprisal':>12} {'Confidence':>12} {'Ratio':>10} {'Log-ratio':>10} {'Winner':>12}")
    print("-" * 76)

    for seed in seeds:
        for task in tasks:
            s_anti = per_sweep[seed]["anti-confident"][task]["speedup"]
            s_conf = per_sweep[seed]["most-confident"][task]["speedup"]

            # Handle zero speedups by using a small epsilon
            if s_conf == 0:
                s_conf = 1e-6
            if s_anti == 0:
                s_anti = 1e-6

            ratio = s_anti / s_conf
            log_ratio = math.log(ratio)
            log_ratios.append(log_ratio)

            if s_anti > s_conf:
                surprisal_wins += 1
                winner = "surprisal"
            elif s_conf > s_anti:
                confidence_wins += 1
                winner = "confidence"
            else:
                ties += 1
                winner = "tie"

            print(f"{seed:>6} {task:>6} {s_anti:>12.3f} {s_conf:>12.3f} {ratio:>10.3f} {log_ratio:>10.3f} {winner:>12}")

    print()

    # --- Paired Wilcoxon signed-rank test ---
    log_ratios_arr = np.array(log_ratios)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(log_ratios_arr, alternative="two-sided")
    print(f"Paired Wilcoxon signed-rank test on log(speedup ratios):")
    print(f"  Test statistic: {wilcoxon_stat:.1f}")
    print(f"  p-value (two-sided): {wilcoxon_p:.4f}")
    print(f"  Median log-ratio: {np.median(log_ratios_arr):.3f}")
    print(f"  Median ratio: {np.exp(np.median(log_ratios_arr)):.3f}x")
    print()

    # --- Binary fast_1 outcomes ---
    print("Binary fast_1 outcomes per task-seed:")
    print(f"{'Seed':>6} {'Task':>6} {'Surprisal':>12} {'Confidence':>12} {'Discordant?':>12}")
    print("-" * 56)

    discordant_surprisal = 0
    discordant_confidence = 0
    concordant = 0

    for seed in seeds:
        for task in tasks:
            f1_anti = per_sweep[seed]["anti-confident"][task]["fast_1"]
            f1_conf = per_sweep[seed]["most-confident"][task]["fast_1"]

            if f1_anti != f1_conf:
                if f1_anti > f1_conf:
                    discordant_surprisal += 1
                    disc = "-> surprisal"
                else:
                    discordant_confidence += 1
                    disc = "-> confidence"
            else:
                concordant += 1
                disc = "concordant"

            print(f"{seed:>6} {task:>6} {f1_anti:>12.1f} {f1_conf:>12.1f} {disc:>12}")

    print()
    print(f"Concordant pairs: {concordant}")
    print(f"Discordant pairs favoring surprisal: {discordant_surprisal}")
    print(f"Discordant pairs favoring confidence: {discordant_confidence}")

    # Exact sign test (one-sided: surprisal > confidence)
    n_discordant = discordant_surprisal + discordant_confidence
    if n_discordant > 0:
        sign_p = stats.binomtest(discordant_surprisal, n_discordant, 0.5, alternative="greater").pvalue
        print(f"Exact sign test (one-sided): p = {sign_p:.3f}")
        print(f"  n_discordant = {n_discordant}, successes = {discordant_surprisal}")
    else:
        print("No discordant pairs for sign test.")

    print()
    print("=" * 60)
    print("SUMMARY FOR PAPER INSERTION:")
    print(f"  Cohen's h = {h:.3f} (medium-to-large effect)")
    print(f"  Wilcoxon: W = {wilcoxon_stat:.1f}, p = {wilcoxon_p:.4f}")
    print(f"  Sign test: {discordant_surprisal}/{n_discordant} discordant favor surprisal, p = {sign_p:.3f}")
    print(f"  Median speedup ratio: {np.exp(np.median(log_ratios_arr)):.1f}x")


if __name__ == "__main__":
    main()
