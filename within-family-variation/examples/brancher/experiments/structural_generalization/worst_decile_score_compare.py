import json
import os

import numpy as np
from scipy import stats

SG = os.path.dirname(__file__)
main = json.load(open(os.path.join(SG, "worst_decile_analysis_results.json")))
sd = json.load(open(os.path.join(SG, "worst_decile_score_discrimination_results.json")))
sd_by_seed = {r["seed"]: r for r in sd}

main = [r for r in main if r.get("root_gap") is not None]
main.sort(key=lambda r: r["ratio"])
decile_n = max(1, round(0.10 * len(main)))
rest_seeds = {r["seed"] for r in main[:-decile_n]}
worst_seeds = {r["seed"] for r in main[-decile_n:]}
ratio_by_seed = {r["seed"]: r["ratio"] for r in main}
gap_by_seed = {r["seed"]: r["root_gap"] for r in main}

PROPS = ["score_std", "score_range", "top1_top2_gap", "top1_top2_gap_normalized"]

print("=== worst decile (n=10) vs rest (n=90) ===\n")
for key in PROPS:
    worst_vals = np.array([sd_by_seed[s][key] for s in worst_seeds], dtype=float)
    rest_vals = np.array([sd_by_seed[s][key] for s in rest_seeds], dtype=float)
    u, p = stats.mannwhitneyu(worst_vals, rest_vals, alternative="two-sided")
    sig = p < 0.05
    print(f"{key}: worst mean={np.nanmean(worst_vals):.4f} median={np.nanmedian(worst_vals):.4f}  "
          f"rest mean={np.nanmean(rest_vals):.4f} median={np.nanmedian(rest_vals):.4f}  "
          f"p={p:.4f}  {'*** SIGNIFICANT ***' if sig else '(not significant)'}")

print("\n=== full-range correlation (n=100) ===\n")
all_seeds = sorted(ratio_by_seed)
ratio = np.array([ratio_by_seed[s] for s in all_seeds])
gap = np.array([gap_by_seed[s] for s in all_seeds])
for key in PROPS:
    x = np.array([sd_by_seed[s][key] for s in all_seeds], dtype=float)
    rp, pp = stats.pearsonr(ratio, x)
    rs, ps = stats.spearmanr(ratio, x)
    rgp, pgp = stats.pearsonr(gap, x)
    rgs, pgs = stats.spearmanr(gap, x)
    print(f"{key}:")
    print(f"  vs ratio:    pearson r={rp:.3f}(p={pp:.3f})  spearman r={rs:.3f}(p={ps:.3f})")
    print(f"  vs root_gap: pearson r={rgp:.3f}(p={pgp:.3f})  spearman r={rgs:.3f}(p={pgs:.3f})")
