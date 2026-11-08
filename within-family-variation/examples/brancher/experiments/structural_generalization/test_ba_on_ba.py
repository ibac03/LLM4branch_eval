"""
ba-on-ba baseline: the LLM-evolved "Original" BA policy (program/indset/program.py,
trained at n=750,affinity=4) run on BA's own n=1000,affinity=4 test tier --
same exact harness/seeds/cap as test_ws_on_ba.py's ws_old-on-ba /
ws_new_warmstart-on-ba cells, so all three are directly comparable. (Not to
be confused with crosstest_rpb_results.json's "ba"/"ba" entry, which is
RPB-with-tuned-params, a different policy, measured at a different cap.)
"""
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, geometric_mean_stable, normalize

SG = os.path.dirname(__file__)
BA_PROGRAM = os.path.join(SG, "..", "..", "program", "indset", "program.py")
SEEDS = list(range(800000000, 800000030))
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
WORKERS = 6
TIME_LIMIT = 900

RESULTS_PATH = os.path.join(SG, "sweep_ba_vs_rr_ws_results.json")


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def solve_one(seed):
    import ecole
    import numpy as np
    import time as _time
    tag = f"[ba-on-ba seed={seed}]"
    print(f"{tag} starting (cap={TIME_LIMIT}s)", flush=True)

    score_function, used_features, function_params, _ = load_program(BA_PROGRAM)
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

    instance = make_ba_instance(seed)
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
    print("=== ba-on-ba baseline (n=1000,affinity=4, 30 seeds, 900s cap) ===", flush=True)
    with Pool(processes=WORKERS) as pool:
        results = pool.map(solve_one, SEEDS)
    results.sort()
    times = [t for _, t, _, _ in results]
    n_optimal = sum(1 for *_, s in results if s == "optimal")
    geo = geometric_mean_stable(times)

    all_results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    all_results.setdefault("1", {}).setdefault("ba", {})["ba"] = {"times": times, "n_optimal": n_optimal}
    tmp = RESULTS_PATH + ".tmp"
    json.dump(all_results, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS_PATH)

    print(f"\nba-on-ba: geomean={geo:.2f}s optimal={n_optimal}/30 max={max(times):.2f}s", flush=True)
    print("\n=== BA_ON_BA_TEST_COMPLETE ===", flush=True)
