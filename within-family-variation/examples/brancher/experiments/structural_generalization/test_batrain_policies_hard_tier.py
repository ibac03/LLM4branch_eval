"""
Test the two BA-difficulty-matched (batrain), BO-tuned policies -- RR n=316
and WS n=300 trainees -- at the SAME hard test tier, same 30-seed block,
same 900s cap, same methodology as sweep_ba_vs_rr_ws_difficulty.py's
checkpoint 1 (RR n=330,degree=3 | WS n=250,k=20,p=0.25). Results are
appended into the SAME sweep_ba_vs_rr_ws_results.json under new
"rr_batrain_tuned" / "ws_batrain_tuned" keys, directly comparable to the
existing ba-on-rr=381.83s/25/30, rr-on-rr(easy-trained)=41.19s/30/30,
ba-on-ws=147.70s/30/30, ws-on-ws(easy-trained)=69.28s/30/30.
"""
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, geometric_mean_stable, normalize
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
PROGRAMS = {
    "rr_batrain_tuned": os.path.join(SG, "rr_evolution_batrain_output", "best", "best_program_tuned.py"),
    "ws_batrain_tuned": os.path.join(SG, "ws_evolution_batrain_output", "best", "best_program_tuned.py"),
}
PAIRS = [("rr_batrain_tuned", "rr"), ("ws_batrain_tuned", "ws")]

SEEDS = list(range(800000000, 800000030))  # identical seed block to the sweep, for direct comparability
CHECKPOINT = {"rr": {"n_nodes": 330, "degree": 3}, "ws": {"n_nodes": 250, "k": 20, "p": 0.25}, "workers": 6}
TIME_LIMIT = 900

RESULTS_PATH = os.path.join(SG, "sweep_ba_vs_rr_ws_results.json")


def make_instance(test_domain, seed):
    p = CHECKPOINT[test_domain]
    if test_domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=p["n_nodes"], degree=p["degree"])
    return generate_ws_instance(seed=seed, n_nodes=p["n_nodes"], k=p["k"], p=p["p"])


def solve_one(args):
    import ecole
    import numpy as np
    import time as _time
    trained_on, test_domain, seed = args
    tag = f"[{trained_on}-on-{test_domain} seed={seed}]"
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

    instance = make_instance(test_domain, seed)
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


def evaluate_cell(trained_on, test_domain):
    tasks = [(trained_on, test_domain, seed) for seed in SEEDS]
    with Pool(processes=CHECKPOINT["workers"]) as pool:
        results = pool.map(solve_one, tasks)
    results.sort()
    times = [t for _, t, _, _ in results]
    n_optimal = sum(1 for *_, s in results if s == "optimal")
    return times, n_optimal


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_cell_result(trained_on, test_domain, times, n_optimal):
    results = load_results()
    results.setdefault("1", {}).setdefault(trained_on, {})[test_domain] = {
        "times": times, "n_optimal": n_optimal,
    }
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


if __name__ == "__main__":
    existing = load_results()
    print("=== batrain (BO-tuned) policies at the hard test tier "
          "(RR n=330,degree=3 | WS n=250,k=20,p=0.25, 30 seeds, 900s cap) ===", flush=True)
    for trained_on, test_domain in PAIRS:
        if "1" in existing and trained_on in existing["1"] and test_domain in existing["1"][trained_on]:
            r = existing["1"][trained_on][test_domain]
            print(f"[resumed] {trained_on}-on-{test_domain}: "
                  f"geomean={geometric_mean_stable(r['times']):.2f}s optimal={r['n_optimal']}/{len(r['times'])}",
                  flush=True)
            continue
        times, n_optimal = evaluate_cell(trained_on, test_domain)
        geo = geometric_mean_stable(times)
        save_cell_result(trained_on, test_domain, times, n_optimal)
        print(f"{trained_on}-on-{test_domain}: geomean={geo:.2f}s optimal={n_optimal}/{len(times)} "
              f"max={max(times):.2f}s", flush=True)

    print("\n=== BATRAIN_HARD_TIER_TEST_COMPLETE ===", flush=True)
