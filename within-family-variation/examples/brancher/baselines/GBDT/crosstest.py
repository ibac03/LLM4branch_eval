"""
Full 3x3 cross-test: GBDT (LightGBM lambdarank) trained per domain x tested
on each domain's held-out TEST set -- the SAME test tiers as the main
LLM4Branch/Evolved-* oracle table (not RPB's recalibrated ones, since GBDT
is a different policy class with its own speed profile -- see the earlier
discussion on why RPB's recalibration doesn't automatically apply here).

BA: n_nodes=1000,affinity=4 (medium), seeds 999999999..1000000028, 3600s cap
RR: n_nodes=250,degree=3, seeds 1000..1029, 120s cap
WS: n_nodes=200,k=20,p=0.25, seeds 200..229, 120s cap

Same incremental-save/resume pattern as crosstest_rpb.py -- writes a cell's
result the moment it's done, and skips already-completed cells on rerun.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "structural_generalization"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, geometric_mean_stable, create_instance, normalize
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

NUM_NODE_FEATURES = 19
NUM_KHALIL_FEATURES = 72
USED_FEATURES = list(range(NUM_NODE_FEATURES + NUM_KHALIL_FEATURES))

BA_SEEDS = list(range(999999999, 999999999 + 30))
RR_SEEDS = list(range(1000, 1030))
WS_SEEDS = list(range(200, 230))
TEST_SEEDS = {"ba": BA_SEEDS, "rr": RR_SEEDS, "ws": WS_SEEDS}
TEST_TIME_LIMIT = {"ba": 3600, "rr": 120, "ws": 120}
TEST_WORKERS = {"ba": 3, "rr": 6, "ws": 6}

HERE = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(HERE, "crosstest_gbdt_results.json")


def make_test_instance(domain, seed):
    if domain == "ba":
        return create_instance("indset", seed, "medium")
    elif domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    else:
        return generate_ws_instance(seed=seed, n_nodes=200, k=20, p=0.25)


def solve_one(args):
    import ecole
    import lightgbm as lgb
    import time as _time
    trained_on, test_domain, seed, model_path = args

    booster = lgb.Booster(model_file=model_path)
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs.set_active_features(list(range(NUM_NODE_FEATURES)))
    flash_khalil_obs.set_active_features(list(range(NUM_KHALIL_FEATURES)))
    scip_params = get_scip_params(time_limit=TEST_TIME_LIMIT[test_domain])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=scip_params,
    )
    env.seed(0)

    instance = make_test_instance(test_domain, seed)
    t0 = _time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    while not done:
        variable_features = np.concatenate(
            (observation["Node"].variable_features[action_set, :],
             observation["Khalil"].features[action_set, :]), axis=1)
        variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
        variable_features = normalize(variable_features.copy(), USED_FEATURES)
        scores = booster.predict(variable_features)
        action = action_set[scores.argmax()]
        observation, action_set, _, done, info = env.step(action)
    proc_time = _time.process_time() - t0
    m = env.model.as_pyscipopt()
    return seed, proc_time, m.getNNodes(), m.getStatus()


def evaluate_cell(trained_on, test_domain, model_path):
    seeds = TEST_SEEDS[test_domain]
    tasks = [(trained_on, test_domain, seed, model_path) for seed in seeds]
    with Pool(processes=TEST_WORKERS[test_domain]) as pool:
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


def save_cell_result(trained_on, tested_on, times, n_optimal):
    results = load_results()
    results.setdefault(trained_on, {})[tested_on] = {"times": times, "n_optimal": n_optimal}
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ba_model", default=os.path.join(HERE, "gbdt_ba.txt"))
    parser.add_argument("--rr_model", default=os.path.join(HERE, "gbdt_rr.txt"))
    parser.add_argument("--ws_model", default=os.path.join(HERE, "gbdt_ws.txt"))
    args = parser.parse_args()

    models = {"ba": args.ba_model, "rr": args.rr_model, "ws": args.ws_model}
    domains = ["ba", "rr", "ws"]

    existing = load_results()
    print("=== 3x3 cross-test (rows=GBDT trained on, cols=tested on) ===", flush=True)
    cell_times = {trained: {} for trained in domains}
    for trained_on in domains:
        for tested_on in domains:
            if trained_on in existing and tested_on in existing[trained_on]:
                times = existing[trained_on][tested_on]["times"]
                n_optimal = existing[trained_on][tested_on]["n_optimal"]
                print(f"GBDT-{trained_on} on {tested_on}: [resumed] "
                      f"geomean={geometric_mean_stable(times):.2f}s optimal={n_optimal}/{len(times)}", flush=True)
                cell_times[trained_on][tested_on] = times
                continue
            times, n_optimal = evaluate_cell(trained_on, tested_on, models[trained_on])
            geo = geometric_mean_stable(times)
            cell_times[trained_on][tested_on] = times
            save_cell_result(trained_on, tested_on, times, n_optimal)
            print(f"GBDT-{trained_on} on {tested_on}: geomean={geo:.2f}s "
                  f"optimal={n_optimal}/{len(times)}", flush=True)

    print("\n=== Per-GBDT pooled 90-instance geomean ===", flush=True)
    pooled_rows = {}
    for trained_on in domains:
        pooled = cell_times[trained_on]["ba"] + cell_times[trained_on]["rr"] + cell_times[trained_on]["ws"]
        pooled_rows[trained_on] = pooled
        print(f"GBDT-{trained_on}: pooled_geomean={geometric_mean_stable(pooled):.2f}s", flush=True)

    oracle = [min(pooled_rows[d][i] for d in domains) for i in range(90)]
    print(f"Oracle (per-instance min across the 3 GBDTs): pooled_geomean={geometric_mean_stable(oracle):.2f}s",
          flush=True)

    win_counts = {d: 0 for d in domains}
    for i in range(90):
        best = min(domains, key=lambda d: pooled_rows[d][i])
        win_counts[best] += 1
    print(f"Win counts across 90 instances: {win_counts}", flush=True)
    print("CROSSTEST_GBDT_DONE", flush=True)
