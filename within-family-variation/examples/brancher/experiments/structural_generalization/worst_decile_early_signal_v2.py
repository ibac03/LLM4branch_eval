"""
#1 (bigger K) + #2 (trajectory, not one snapshot) in a single probe: run the
real policy out to CHECKPOINTS[-1] nodes, recording the live gap at each
checkpoint along the way (no extra solving cost over just running to the
largest checkpoint once). Lets us test whether looking further than the
original K=50 helps, and whether the gap's trajectory (how fast it's
closing) carries more signal than any single snapshot.
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
RESULTS_PATH = os.path.join(SG, "worst_decile_early_signal_v2_results.json")
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
CHECKPOINTS = [10, 25, 50, 100, 200]
SAFETY_TIME_LIMIT = 300


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def early_signal_v2(seed):
    import ecole
    import time as _time
    print(f"[seed={seed}] running to {CHECKPOINTS[-1]} nodes with checkpoints {CHECKPOINTS}", flush=True)

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

    checkpoint_data = {}  # node_target -> (nodes_actual, cpu_time, gap, finished)
    next_ckpt_idx = 0
    K_final = CHECKPOINTS[-1]

    while not done and nodes < K_final:
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

        while next_ckpt_idx < len(CHECKPOINTS) and nodes >= CHECKPOINTS[next_ckpt_idx]:
            checkpoint_data[CHECKPOINTS[next_ckpt_idx]] = {
                "nodes_actual": nodes, "cpu_time": _time.process_time() - t0,
                "gap": m.getGap(), "finished": bool(done),
            }
            next_ckpt_idx += 1

    # if done fired before reaching some/all checkpoints, backfill with final state
    final_cpu = _time.process_time() - t0
    final_gap = m.getGap()
    for ck in CHECKPOINTS:
        if ck not in checkpoint_data:
            checkpoint_data[ck] = {
                "nodes_actual": nodes, "cpu_time": final_cpu, "gap": final_gap, "finished": bool(done),
            }

    rec = {"seed": seed}
    for ck in CHECKPOINTS:
        d = checkpoint_data[ck]
        rec[f"gap_at_{ck}"] = d["gap"]
        rec[f"cpu_time_at_{ck}"] = d["cpu_time"]
        rec[f"nodes_actual_at_{ck}"] = d["nodes_actual"]
        rec[f"finished_by_{ck}"] = d["finished"]

    print(f"[seed={seed}] done: " + " ".join(f"gap@{ck}={checkpoint_data[ck]['gap']:.4f}" for ck in CHECKPOINTS),
          flush=True)
    return rec


if __name__ == "__main__":
    from multiprocessing import Pool
    main = json.load(open(MAIN_RESULTS))
    seeds = sorted(r["seed"] for r in main)
    print(f"=== early-signal v2 (checkpoints {CHECKPOINTS}): {len(seeds)} BA instances ===", flush=True)
    with Pool(processes=6) as pool:
        records = pool.map(early_signal_v2, seeds)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== EARLY_SIGNAL_V2_COMPLETE ===", flush=True)
