"""
Test the WS warm-start follow-up policy (0.7994 training-tier stage1_score,
evolved from the 0.7326 BO-tuned seed for 200 more iterations, gate
unchanged at 0.85 -- never cleared it) at the same hard test tier as
everything else in this investigation: n=250,k=20,p=0.25, 30 seeds, 900s
cap. Directly comparable to the existing results in
sweep_ba_vs_rr_ws_results.json: ba-on-ws=147.70s/30/30,
ws-on-ws(easy-trained)=69.28s/30/30, ws_batrain_tuned-on-ws=72.08s/30/30.
"""
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, geometric_mean_stable, normalize
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
PROGRAM = os.path.join(SG, "ws_evolution_batrain_warmstart_output", "best", "best_program.py")
SEEDS = list(range(800000000, 800000030))
PARAMS_TEST = {"n_nodes": 250, "k": 20, "p": 0.25}
WORKERS = 6
TIME_LIMIT = 900

RESULTS_PATH = os.path.join(SG, "sweep_ba_vs_rr_ws_results.json")


def solve_one(seed):
    import ecole
    import numpy as np
    import time as _time
    tag = f"[ws_batrain_warmstart-on-ws seed={seed}]"
    print(f"{tag} starting (cap={TIME_LIMIT}s)", flush=True)

    score_function, used_features, function_params, _ = load_program(PROGRAM)
    score_function = partial(score_function, params=function_params)

    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=TIME_LIMIT),
    )
    env.seed(0)

    instance = generate_ws_instance(seed=seed, n_nodes=PARAMS_TEST["n_nodes"],
                                     k=PARAMS_TEST["k"], p=PARAMS_TEST["p"])
    t0 = _time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    while not done:
        variable_features = np.concatenate(
            (observation["Node"].variable_features[action_set, :],
             observation["Khalil"].features[action_set, :]), axis=1)
        variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
        variable_features = normalize(variable_features, used_features)
        scores = score_function(variable_features)
        action = action_set[scores.argmax()]
        observation, action_set, _, done, info = env.step(action)
    proc_time = _time.process_time() - t0
    m = env.model.as_pyscipopt()
    status = m.getStatus()
    print(f"{tag} done: time={proc_time:.2f}s nodes={m.getNNodes()} status={status}", flush=True)
    return seed, proc_time, m.getNNodes(), status


if __name__ == "__main__":
    print("=== ws_batrain_warmstart-on-ws at hard test tier "
          "(n=250,k=20,p=0.25, 30 seeds, 900s cap) ===", flush=True)
    with Pool(processes=WORKERS) as pool:
        results = pool.map(solve_one, SEEDS)
    results.sort()
    times = [t for _, t, _, _ in results]
    n_optimal = sum(1 for *_, s in results if s == "optimal")
    geo = geometric_mean_stable(times)

    all_results = {}
    if os.path.exists(RESULTS_PATH):
        all_results = json.load(open(RESULTS_PATH))
    all_results.setdefault("1", {}).setdefault("ws_batrain_warmstart", {})["ws"] = {
        "times": times, "n_optimal": n_optimal,
    }
    tmp = RESULTS_PATH + ".tmp"
    json.dump(all_results, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS_PATH)

    print(f"\nws_batrain_warmstart-on-ws: geomean={geo:.2f}s optimal={n_optimal}/{len(times)} "
          f"max={max(times):.2f}s", flush=True)
    print("\n=== WS_WARMSTART_HARD_TIER_TEST_COMPLETE ===", flush=True)
