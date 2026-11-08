"""
Compare the worst-decile-by-ratio BA instances against the rest on the 4
structural properties, per worst_decile_analysis.py's collected data.
"worst" = policy did WORST relative to RPB = HIGHEST ratio (sorted
ascending, worst decile is the top 10% of ratio values, at the tail).
"""
import json
import os
import statistics
import sys

import numpy as np
from scipy import stats

SG = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SG, "worst_decile_analysis_results.json")

PROPERTIES = [
    ("degree_cv", "Degree coefficient of variation"),
    ("clustering_coefficient", "Clustering coefficient"),
    ("n_edges", "Graph size (n_edges; n_nodes is constant=1000 by construction)"),
    ("root_gap", "Root LP gap"),
]


def main():
    records = json.load(open(RESULTS_PATH))
    records = [r for r in records if r.get("root_gap") is not None]
    n = len(records)
    decile_n = max(1, round(0.10 * n))

    records.sort(key=lambda r: r["ratio"])
    rest = records[:-decile_n]
    worst = records[-decile_n:]

    print(f"n={n} instances (dropped {len(json.load(open(RESULTS_PATH))) - n} with unresolved root_gap)", flush=True)
    print(f"worst decile = top {decile_n} by ratio (policy_time/rpb_time)\n", flush=True)

    print("=== ratio itself ===", flush=True)
    print(f"  rest  : min={min(r['ratio'] for r in rest):.3f} "
          f"median={statistics.median(r['ratio'] for r in rest):.3f} "
          f"max={max(r['ratio'] for r in rest):.3f}", flush=True)
    print(f"  worst : {[round(r['ratio'], 2) for r in worst]}", flush=True)
    print(f"          seeds: {[r['seed'] for r in worst]}", flush=True)
    n_policy_to_status = {}
    for r in worst:
        n_policy_to_status[r["policy_status"]] = n_policy_to_status.get(r["policy_status"], 0) + 1
    print(f"  worst-decile policy statuses: {n_policy_to_status}\n", flush=True)

    any_significant = False
    for key, label in PROPERTIES:
        worst_vals = np.array([r[key] for r in worst], dtype=float)
        rest_vals = np.array([r[key] for r in rest], dtype=float)
        u_stat, p_value = stats.mannwhitneyu(worst_vals, rest_vals, alternative="two-sided")
        sig = p_value < 0.05
        any_significant = any_significant or sig
        print(f"=== {label} ===", flush=True)
        print(f"  worst decile (n={len(worst_vals)}): mean={worst_vals.mean():.4f} "
              f"median={np.median(worst_vals):.4f} std={worst_vals.std():.4f}", flush=True)
        print(f"  rest        (n={len(rest_vals)}):  mean={rest_vals.mean():.4f} "
              f"median={np.median(rest_vals):.4f} std={rest_vals.std():.4f}", flush=True)
        print(f"  Mann-Whitney U p-value: {p_value:.4f}  "
              f"{'*** SIGNIFICANT (p<0.05) ***' if sig else '(not significant)'}\n", flush=True)

    print("=== VERDICT ===", flush=True)
    if any_significant:
        print("At least one structural property differs significantly between the worst decile "
              "and the rest -- there IS a detectable pattern in which BA instances (drawn from the "
              "standard benchmark itself) the policy struggles on. This would support evolving a "
              "specialist/selector: the failure mode is predictable from instance features.", flush=True)
    else:
        print("None of the 4 properties differ significantly -- the worst decile looks structurally "
              "indistinguishable from the rest. This suggests the slowdown is not predictable from "
              "these instance-level features; it may instead be branch-and-bound's inherent "
              "chaotic sensitivity to tie-breaking/search order. No pattern to select on here.", flush=True)


if __name__ == "__main__":
    main()
