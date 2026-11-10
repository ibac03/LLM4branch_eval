"""Cheap features for each instance: graph, root node, and the first 50 nodes of the original policy.

    python scripts/probe_features.py --level medium --seeds 999999999:1300 --cores 104-127 --out runs/task4_medium/features.csv

The probe runs a policy (default: the original LLM4Branch Indset policy, same solve as the official
tester) and stops at 50 nodes. If the router keeps the original policy, the probe is the start of that solve (no cost).
Feature groups (prefix): g_ = graph (before the solve), r_ = root node, kN_ = after N nodes.
"""
import argparse
import itertools
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "LLM4Branch", "examples", "brancher"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
CHECKPOINTS = [1, 5, 10, 25, 50]


def graph_features(instance):
    import networkx as nx
    m = instance.as_pyscipopt()
    idx = {v.name: i for i, v in enumerate(m.getVars())}
    G = nx.Graph()
    G.add_nodes_from(range(len(idx)))
    sizes = []
    for c in m.getConss():
        members = [idx[name] for name in m.getValsLinear(c)]
        sizes.append(len(members))
        G.add_edges_from(itertools.combinations(members, 2))
    deg = np.array([d for _, d in G.degree()])
    core = np.array(list(nx.core_number(G).values()))
    return {"g_edges": G.number_of_edges(), "g_degree_cv": deg.std() / deg.mean(), "g_degree_max": deg.max(),
            "g_clustering": nx.average_clustering(G), "g_triangles": sum(nx.triangles(G).values()) / 3,
            "g_assortativity": nx.degree_assortativity_coefficient(G), "g_core_max": core.max(), "g_core_mean": core.mean(),
            "g_n_cliques": len(sizes), "g_clique_size_mean": np.mean(sizes), "g_clique_size_max": np.max(sizes)}


def snapshot(m, t0, dual_root, prefix):
    opened = m.getNLeaves() + m.getNChildren() + m.getNSiblings()
    return {f"{prefix}_gap": m.getGap(), f"{prefix}_time": time.process_time() - t0,
            f"{prefix}_dual_move": abs(m.getDualbound() - dual_root) / (abs(dual_root) + 1e-9),
            f"{prefix}_open_nodes": opened, f"{prefix}_depth": m.getDepth(),
            f"{prefix}_lp_iters": m.getNLPIterations(), f"{prefix}_n_sols": m.getNBestSolsFound()}


def probe(seed, level, policy, env_seed, graph):
    import ecole
    from instances import make_instance
    from utils.utils import get_scip_params, load_program, normalize
    rec = {"seed": seed}
    try:
        if graph:
            rec.update(graph_features(make_instance(level, seed)))
        score_function, used, params, _ = load_program(os.path.join(ROOT, "policies", policy + ".py"))
        score_function = partial(score_function, params=params)
        khalil, node = ecole.observation.Khalil2016Flash(), ecole.observation.NodeBipartiteFlash()
        khalil.set_active_features([f - 19 for f in used if f >= 19])
        node.set_active_features([f for f in used if f < 19])
        env = ecole.environment.Branching(observation_function={"Node": node, "Khalil": khalil},
                                          scip_params=get_scip_params(time_limit=600))
        env.seed(env_seed)
        t0 = time.process_time()
        obs, action_set, _, done, _ = env.reset(make_instance(level, seed))
        m = env.model.as_pyscipopt()
        dual_root = m.getDualbound()
        rec.update({"r_time": time.process_time() - t0, "r_gap": m.getGap(), "r_dual": dual_root,
                    "r_primal": m.getPrimalbound(), "r_lp_iters": m.getNLPIterations(),
                    "r_n_cands": 0 if done else len(action_set), "r_n_sols": m.getNBestSolsFound()})
        todo, n_cands, margins, first = list(CHECKPOINTS), [], [], True
        while not done and todo:
            n = m.getNNodes()
            while todo and n >= todo[0]:
                rec.update(snapshot(m, t0, dual_root, f"k{todo.pop(0)}"))
            if not todo:
                break
            f = np.concatenate((obs["Node"].variable_features[action_set, :], obs["Khalil"].features[action_set, :]), axis=1)
            f[:, used] = np.nan_to_num(f[:, used], nan=0.0, posinf=0.0, neginf=0.0)
            if first and policy == "orig":  # raw candidate features at the root: obj coef, reduced cost, LP fractional part, constraint count
                frac = f[:, 9]
                rec.update({"r_frac_mean": frac.mean(), "r_frac_std": frac.std(), "r_frac_near_half": (np.abs(frac - 0.5) < 0.1).mean(),
                            "r_redcost_std": f[:, 7].std(), "r_cons_mean": f[:, 22].mean(), "r_cons_max": f[:, 22].max()})
                first = False
            scores = score_function(normalize(f, used))
            top = np.sort(scores)
            n_cands.append(len(action_set))
            margins.append(top[-1] - top[-2] if len(top) > 1 else 0.0)
            obs, action_set, _, done, _ = env.step(action_set[scores.argmax()])
            m = env.model.as_pyscipopt()
        for k in todo:  # solved before this checkpoint
            rec.update(snapshot(m, t0, dual_root, f"k{k}"))
        rec.update({"k50_cands_mean": np.mean(n_cands) if n_cands else 0, "k50_cands_last": n_cands[-1] if n_cands else 0,
                    "k50_margin_mean": np.mean(margins) if margins else 0,
                    "finished_in_probe": int(bool(done)), "nodes_reached": m.getNNodes()})
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="medium")
    p.add_argument("--seeds", required=True, help="START:N")
    p.add_argument("--cores", default="104-127")
    p.add_argument("--out", required=True)
    p.add_argument("--policy", default="orig", help="policy that runs the probe")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--no-graph", action="store_true", help="skip the graph features")
    a = p.parse_args()
    start, n = map(int, a.seeds.split(":"))
    lo, hi = map(int, a.cores.split("-"))
    os.sched_setaffinity(0, range(lo, hi + 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t = time.time()
    with Pool(hi - lo + 1) as pool:
        recs = pool.map(partial(probe, level=a.level, policy=a.policy, env_seed=a.env_seed, graph=not a.no_graph), range(start, start + n), chunksize=2)
    pd.DataFrame(recs).sort_values("seed").to_csv(a.out, index=False)
    print(f"PROBE_DONE {len(recs)} instances in {(time.time() - t) / 60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
