"""
Follow-up #4: hub REDUNDANCY, not hub amount. Degree CV asks "how uneven
are the degrees overall" -- already tested, no signal. This asks a
different question: is there one dominant hub, or several competing ones
of similar size? Same graph reconstruction as worst_decile_analysis.py
(clique-constraint union -> edges), no solving needed. Runs on whatever
seeds are already in worst_decile_analysis_results.json, so it naturally
covers 400 now and 1300 once the extension finishes.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SG = os.path.dirname(__file__)
MAIN_RESULTS = os.path.join(SG, "worst_decile_analysis_results.json")
RESULTS_PATH = os.path.join(SG, "worst_decile_hub_redundancy_results.json")
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def hub_redundancy_stats(seed):
    import networkx as nx
    instance = make_ba_instance(seed)
    m = instance.as_pyscipopt()
    n_vars = len(m.getVars())
    name_to_idx = {v.name: i for i, v in enumerate(m.getVars())}
    G = nx.Graph()
    G.add_nodes_from(range(n_vars))
    for c in m.getConss():
        coefs = m.getValsLinear(c)
        idxs = [name_to_idx[name] for name in coefs.keys()]
        for a, b in itertools.combinations(idxs, 2):
            G.add_edge(a, b)

    degs = sorted((d for _, d in G.degree()), reverse=True)
    top1, top2, top3 = degs[0], degs[1], degs[2]
    top5_sum = sum(degs[:5])
    total_degree = sum(degs)
    top1_top2_ratio = top1 / top2 if top2 > 0 else float("nan")
    top5_share = top5_sum / total_degree
    threshold = 0.8 * top1
    n_near_top = sum(1 for d in degs if d >= threshold)

    rec = {
        "seed": seed, "top1_degree": top1, "top2_degree": top2, "top3_degree": top3,
        "top1_top2_degree_ratio": top1_top2_ratio, "top5_degree_share": top5_share,
        "n_near_top_hubs": n_near_top,
    }
    print(f"[seed={seed}] top1={top1} top2={top2} ratio={top1_top2_ratio:.3f} "
          f"top5_share={top5_share:.4f} n_near_top={n_near_top}", flush=True)
    return rec


if __name__ == "__main__":
    from multiprocessing import Pool
    main = json.load(open(MAIN_RESULTS))
    seeds = sorted(r["seed"] for r in main)
    print(f"=== hub redundancy stats: {len(seeds)} BA instances (no solving needed) ===", flush=True)
    with Pool(processes=6) as pool:
        records = pool.map(hub_redundancy_stats, seeds)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== HUB_REDUNDANCY_COMPLETE ===", flush=True)
