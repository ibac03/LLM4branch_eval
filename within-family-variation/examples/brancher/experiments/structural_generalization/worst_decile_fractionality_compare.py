"""
Join worst_decile_fractionality_results.json onto the SAME worst-decile /
rest split used in worst_decile_compare.py (by seed), and test whether the
pre-solve-computable root fractionality proxy differs.
"""
import json
import os

import numpy as np
from scipy import stats

SG = os.path.dirname(__file__)
main = json.load(open(os.path.join(SG, "worst_decile_analysis_results.json")))
frac = json.load(open(os.path.join(SG, "worst_decile_fractionality_results.json")))
frac_by_seed = {r["seed"]: r for r in frac}

main = [r for r in main if r.get("root_gap") is not None]
main.sort(key=lambda r: r["ratio"])
decile_n = max(1, round(0.10 * len(main)))
rest_seeds = {r["seed"] for r in main[:-decile_n]}
worst_seeds = {r["seed"] for r in main[-decile_n:]}

PROPS = [("mean_root_fractionality", "Mean root LP fractionality"),
         ("root_fractional_share", "Share of root candidates non-integral")]

for key, label in PROPS:
    worst_vals = np.array([frac_by_seed[s][key] for s in worst_seeds])
    rest_vals = np.array([frac_by_seed[s][key] for s in rest_seeds])
    u, p = stats.mannwhitneyu(worst_vals, rest_vals, alternative="two-sided")
    sig = p < 0.05
    print(f"=== {label} (pre-solve proxy) ===")
    print(f"  worst decile (n={len(worst_vals)}): mean={worst_vals.mean():.4f} median={np.median(worst_vals):.4f}")
    print(f"  rest        (n={len(rest_vals)}):  mean={rest_vals.mean():.4f} median={np.median(rest_vals):.4f}")
    print(f"  Mann-Whitney U p-value: {p:.4f}  {'*** SIGNIFICANT ***' if sig else '(not significant)'}\n")
