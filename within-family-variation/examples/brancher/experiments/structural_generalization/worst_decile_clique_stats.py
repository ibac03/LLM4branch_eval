"""
Follow-up #2 to worst_decile_analysis.py: neither degree CV nor clustering
(generic graph properties) explained root_gap. This tests a property tied
directly to HOW THIS SOLVER ACTUALLY WORKS: the independent-set MILP uses
one constraint per (maximal) clique, not per edge (reverse-engineered and
validated earlier). The root LP relaxation's tightness is mechanically a
function of the clique structure -- a few large, dominant cliques give the
relaxation more to grab onto than the same edges spread across many small
ones. No solving needed here, just re-reading each instance's constraints
(cheap/fast) -- same 100 seeds as every other check in this analysis.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SG = os.path.dirname(__file__)
N_INSTANCES = 100
SEEDS = list(range(800000000, 800000000 + N_INSTANCES))
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
RESULTS_PATH = os.path.join(SG, "worst_decile_clique_stats_results.json")


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def clique_stats(seed):
    instance = make_ba_instance(seed)
    m = instance.as_pyscipopt()
    sizes = []
    for c in m.getConss():
        coefs = m.getValsLinear(c)
        sizes.append(len(coefs))
    sizes = np.array(sizes)
    sizes_sorted = np.sort(sizes)[::-1]
    max1 = int(sizes_sorted[0])
    max2 = int(sizes_sorted[1]) if len(sizes_sorted) > 1 else 0
    rec = {
        "seed": seed,
        "n_cliques": len(sizes),
        "max_clique_size": max1,
        "second_max_clique_size": max2,
        "top1_top2_ratio": float(max1 / max2) if max2 > 0 else float("nan"),
        "mean_clique_size": float(sizes.mean()),
        "clique_size_cv": float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else float("nan"),
    }
    print(f"[seed={seed}] n_cliques={rec['n_cliques']} max={max1} 2nd_max={max2} "
          f"mean={rec['mean_clique_size']:.2f} cv={rec['clique_size_cv']:.3f}", flush=True)
    return rec


if __name__ == "__main__":
    from multiprocessing import Pool
    print(f"=== clique-size stats: {N_INSTANCES} BA instances (no solving needed) ===", flush=True)
    with Pool(processes=6) as pool:
        records = pool.map(clique_stats, SEEDS)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== CLIQUE_STATS_COMPLETE ===", flush=True)
