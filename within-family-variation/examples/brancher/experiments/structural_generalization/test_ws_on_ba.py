"""
Cross-domain test: the OLD easy-trained WS policy and the NEW warm-started
WS policy, both run on BA's own test instances (n=1000,affinity=4 -- BA's
test-tier convention used elsewhere in this investigation, one tier above
BA's own n=750 training size). Same harness/seeds/cap as every other
number in this session (30-seed block, 900s cap, 6 workers) so this is a
clean, directly-comparable fresh measurement rather than reusing an older
script's possibly-different time-cap convention.
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
PROGRAMS = {
    "ws_old": os.path.join(SG, "ws_evolution_output", "best", "best_program.py"),
    "ws_new_warmstart": os.path.join(SG, "ws_evolution_batrain_warmstart_output", "best", "best_program.py"),
}
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


def solve_one(args):
    import ecole
    import numpy as np
    import time as _time
    trained_on, seed = args
    tag = f"[{trained_on}-on-ba seed={seed}]"
    print(f"{tag} starting (cap={TIME_LIMIT}s)", flush=True)

    score_function, used_features, function_params, _ = load_program(PROGRAMS[trained_on])
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


def evaluate_cell(trained_on):
    tasks = [(trained_on, seed) for seed in SEEDS]
    with Pool(processes=WORKERS) as pool:
        results = pool.map(solve_one, tasks)
    results.sort()
    times = [t for _, t, _, _ in results]
    n_optimal = sum(1 for *_, s in results if s == "optimal")
    return times, n_optimal


if __name__ == "__main__":
    print("=== WS (old vs new) on BA's own test tier (n=1000,affinity=4, 30 seeds, 900s cap) ===", flush=True)
    all_results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    summary = {}
    for trained_on in ["ws_old", "ws_new_warmstart"]:
        times, n_optimal = evaluate_cell(trained_on)
        geo = geometric_mean_stable(times)
        summary[trained_on] = (geo, n_optimal, max(times))
        all_results.setdefault("1", {}).setdefault(trained_on, {})["ba"] = {
            "times": times, "n_optimal": n_optimal,
        }
        tmp = RESULTS_PATH + ".tmp"
        json.dump(all_results, open(tmp, "w"), indent=2)
        os.replace(tmp, RESULTS_PATH)
        print(f"{trained_on}-on-ba: geomean={geo:.2f}s optimal={n_optimal}/{len(times)} max={max(times):.2f}s",
              flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for trained_on, (geo, n_optimal, mx) in summary.items():
        print(f"{trained_on}-on-ba: geomean={geo:.2f}s optimal={n_optimal}/30 max={mx:.2f}s", flush=True)
    print("\n=== WS_ON_BA_TEST_COMPLETE ===", flush=True)
