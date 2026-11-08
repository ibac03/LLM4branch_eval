"""
Follow-up #5 (the promising one): a few seconds of REAL early search,
instead of a purely pre-solve static feature. Run the actual BA policy for
a short, fixed burst -- the first K branch-and-bound nodes -- then look at
how far the optimality gap has closed and how expensive those K nodes
were. This is a tiny fraction of a full solve (worst instances run
hundreds of thousands of nodes; K=50 is essentially free by comparison),
but it's watching the real mechanism instead of guessing from static graph
structure, which is the class of feature the MIP-difficulty-prediction
literature generally finds most informative.
"""
import json
import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, normalize

SG = os.path.dirname(__file__)
BA_PROGRAM = os.path.join(SG, "..", "..", "program", "indset", "program.py")
MAIN_RESULTS = os.path.join(SG, "worst_decile_analysis_results.json")
RESULTS_PATH = os.path.join(SG, "worst_decile_early_signal_results.json")
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
K_NODES = 50
SAFETY_TIME_LIMIT = 120  # just a safety net; the real cutoff is K_NODES


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def early_signal(seed):
    import ecole
    import time as _time
    print(f"[seed={seed}] running first {K_NODES} nodes", flush=True)

    score_function, used_features, function_params, _ = load_program(BA_PROGRAM)
    score_function = partial(score_function, params=function_params)

    instance = make_ba_instance(seed)
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=SAFETY_TIME_LIMIT))
    env.seed(0)

    t0 = _time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    m = env.model.as_pyscipopt()
    nodes = m.getNNodes()
    while not done and nodes < K_NODES:
        variable_features = np.concatenate(
            (observation["Node"].variable_features[action_set, :],
             observation["Khalil"].features[action_set, :]), axis=1)
        variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
        variable_features = normalize(variable_features, used_features)
        scores = score_function(variable_features)
        action = action_set[scores.argmax()]
        observation, action_set, _, done, info = env.step(action)
        m = env.model.as_pyscipopt()
        nodes = m.getNNodes()
    cpu_time = _time.process_time() - t0

    gap = m.getGap()
    rec = {
        "seed": seed, "nodes_reached": nodes, "finished_early": bool(done),
        "cpu_time_to_k_nodes": cpu_time, "gap_at_k_nodes": gap,
        "nodes_per_second": nodes / cpu_time if cpu_time > 0 else float("nan"),
    }
    print(f"[seed={seed}] done: nodes={nodes} finished={done} time={cpu_time:.3f}s "
          f"gap={gap:.5f} nodes/s={rec['nodes_per_second']:.1f}", flush=True)
    return rec


if __name__ == "__main__":
    from multiprocessing import Pool
    main = json.load(open(MAIN_RESULTS))
    seeds = sorted(r["seed"] for r in main)
    print(f"=== early-search signal (first {K_NODES} nodes): {len(seeds)} BA instances ===", flush=True)
    with Pool(processes=6) as pool:
        records = pool.map(early_signal, seeds)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== EARLY_SIGNAL_COMPLETE ===", flush=True)
