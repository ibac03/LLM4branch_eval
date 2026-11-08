"""
Does the BA "Original" policy's worst-case slowdown vs RPB correlate with
graph structure WITHIN the standard BA benchmark itself (not just "RR/WS
are a different graph family")?

For 100 BA instances (n=1000,affinity=4 -- same test tier as every other
BA number in this session), same seed solved by BOTH the policy and RPB
(one task per seed, guaranteeing the pairing), compute:
  - ratio = policy_time / rpb_time
  - degree coefficient of variation, average clustering coefficient,
    n_nodes/n_edges, root LP gap
-- reconstructing the graph from the clique-constraint MILP (ecole's
IndependentSetGenerator does NOT emit one constraint per edge; it emits a
clique cover, so edges = union of all pairs within each constraint --
validated against the theoretical BA edge count affinity*(n-affinity)
before this run).

Worst decile = bottom 10% by ratio... wait: HIGHEST ratio = policy did
WORST relative to RPB, so "worst decile" = the top 10 by ratio value once
sorted ascending. Compared against the rest via descriptive stats +
Mann-Whitney U (robust to the small, non-normal worst-decile sample).
"""
import itertools
import json
import os
import sys
import time as _time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params

SG = os.path.dirname(__file__)
BA_PROGRAM = os.path.join(SG, "..", "..", "program", "indset", "program.py")
N_INSTANCES = 100
SEEDS = list(range(800000000, 800000000 + N_INSTANCES))
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
TIME_LIMIT = 900
WORKERS = 6

RESULTS_PATH = os.path.join(SG, "worst_decile_analysis_results.json")


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def graph_stats(instance):
    import networkx as nx
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
    degs = np.array([d for _, d in G.degree()])
    degree_cv = float(degs.std() / degs.mean()) if degs.mean() > 0 else float("nan")
    clustering = float(nx.average_clustering(G))
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "degree_cv": degree_cv,
        "clustering_coefficient": clustering,
    }


def solve(instance, branch_method, score_function=None, used_features=None):
    import ecole
    import numpy as _np
    sys.path.insert(0, SG)
    from utils.utils import normalize
    scip_params = {**get_scip_params(time_limit=TIME_LIMIT)}
    if branch_method == "relpscost":
        scip_params["branching/relpscost/priority"] = 9999999
        env = ecole.environment.Configuring(scip_params=scip_params)
        env.seed(0)
        t0 = _time.process_time()
        env.reset(instance)
        env.step({})
        proc_time = _time.process_time() - t0
    else:
        flash_khalil_obs = ecole.observation.Khalil2016Flash()
        flash_node_obs = ecole.observation.NodeBipartiteFlash()
        flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
        flash_node_obs.set_active_features([f for f in used_features if f < 19])
        env = ecole.environment.Branching(
            observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
            scip_params=scip_params)
        env.seed(0)
        t0 = _time.process_time()
        observation, action_set, _, done, info = env.reset(instance)
        while not done:
            variable_features = _np.concatenate(
                (observation["Node"].variable_features[action_set, :],
                 observation["Khalil"].features[action_set, :]), axis=1)
            variable_features = _np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
            variable_features = normalize(variable_features, used_features)
            scores = score_function(variable_features)
            action = action_set[scores.argmax()]
            observation, action_set, _, done, info = env.step(action)
        proc_time = _time.process_time() - t0

    m = env.model.as_pyscipopt()
    return {
        "time": proc_time,
        "status": m.getStatus(),
        "objval": m.getObjVal() if m.getStatus() in ("optimal", "timelimit", "bestsollimit") else None,
        "dualbound_root": m.getDualboundRoot(),
    }


def run_one_seed(seed):
    from functools import partial
    print(f"[seed={seed}] starting (cap={TIME_LIMIT}s)", flush=True)

    inst_for_stats = make_ba_instance(seed)
    stats = graph_stats(inst_for_stats)

    score_function, used_features, function_params, _ = load_program(BA_PROGRAM)
    score_function = partial(score_function, params=function_params)

    inst_policy = make_ba_instance(seed)
    pol = solve(inst_policy, "evolve", score_function, used_features)

    inst_rpb = make_ba_instance(seed)
    rpb = solve(inst_rpb, "relpscost")

    # "true optimal": prefer RPB's (it's the faster/more-reliable solver here);
    # fall back to the policy's if RPB itself didn't close the gap.
    true_opt = rpb["objval"] if rpb["status"] == "optimal" else pol["objval"]
    root_gap = None
    if true_opt is not None:
        root_gap = abs(pol["dualbound_root"] - true_opt) / (abs(true_opt) + 1e-6)

    ratio = pol["time"] / rpb["time"] if rpb["time"] > 0 else float("inf")
    record = {
        "seed": seed,
        "policy_time": pol["time"], "policy_status": pol["status"],
        "rpb_time": rpb["time"], "rpb_status": rpb["status"],
        "ratio": ratio,
        "root_gap": root_gap,
        "dualbound_root_policy_run": pol["dualbound_root"],
        "dualbound_root_rpb_run": rpb["dualbound_root"],
        **stats,
    }
    print(f"[seed={seed}] done: policy={pol['time']:.2f}s({pol['status']}) "
          f"rpb={rpb['time']:.2f}s({rpb['status']}) ratio={ratio:.3f} "
          f"degree_cv={stats['degree_cv']:.3f} clustering={stats['clustering_coefficient']:.4f} "
          f"n_edges={stats['n_edges']} root_gap={root_gap}", flush=True)
    return record


if __name__ == "__main__":
    print(f"=== worst-decile structural analysis: {N_INSTANCES} BA instances "
          f"(n=1000,affinity=4), policy vs RPB, same seed each ===", flush=True)
    with Pool(processes=WORKERS) as pool:
        records = pool.map(run_one_seed, SEEDS)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== WORST_DECILE_DATA_COMPLETE ===", flush=True)
