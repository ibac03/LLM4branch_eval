"""
Re-evaluate the 5 existing RPB tunings (ba/rr/ws/mixed/default, run1's
params -- the canonical small-budget tuning, validated by the bigscale
check to not need a bigger budget) on the RECALIBRATED RR/WS test tiers:

  RR: n_nodes=310, degree=3   (was n_nodes=250 -- old tier was too easy for
                                RPB specifically, avg ~3.5-5s; new tier
                                avg=28.70s, all-optimal, 8/8 verified)
  WS: n_nodes=200, k=20, p=0.3 (was p=0.25 -- old tier avg ~9-13s; new tier
                                avg=30.20s, all-optimal, 4/4 verified)

BA is UNCHANGED (n_nodes=1000,affinity=4) -- no need to re-run it, its
existing results from crosstest_rpb_results.json are reused as-is.

Same seeds as before (RR: 1000-1029, WS: 200-229) for continuity with
every other row/domain in the oracle table.
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, geometric_mean_stable
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

RR_SEEDS = list(range(1000, 1030))
WS_SEEDS = list(range(200, 230))
TEST_SEEDS = {"rr": RR_SEEDS, "ws": WS_SEEDS}
TEST_TIME_LIMIT = 120  # unchanged, matches the rest of this pipeline's RR/WS convention
TEST_WORKERS = 6

SG = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SG, "crosstest_rpb_results_recalibrated.json")

# New, recalibrated instance params (only RR/WS change; BA is untouched)
RR_N_NODES, RR_DEGREE = 310, 3
WS_N_NODES, WS_K, WS_P = 200, 20, 0.3


def make_test_instance(test_domain, seed):
    if test_domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=RR_N_NODES, degree=RR_DEGREE)
    else:
        return generate_ws_instance(seed=seed, n_nodes=WS_N_NODES, k=WS_K, p=WS_P)


def solve_one(args):
    import ecole
    tuned_domain, test_domain, seed, params = args
    instance = make_test_instance(test_domain, seed)
    scip_params = {
        **get_scip_params(time_limit=TEST_TIME_LIMIT),
        "branching/relpscost/priority": 9999999,
    }
    for name, value in params.items():
        scip_params[f"branching/relpscost/{name}"] = float(value)

    env = ecole.environment.Configuring(scip_params=scip_params)
    env.seed(0)
    import time as _time
    t0 = _time.process_time()
    env.reset(instance)
    _, _, _, _, _ = env.step({})
    proc_time = _time.process_time() - t0
    m = env.model.as_pyscipopt()
    return seed, proc_time, m.getNNodes(), m.getStatus()


def evaluate_cell(tuned_domain, test_domain, params):
    seeds = TEST_SEEDS[test_domain]
    tasks = [(tuned_domain, test_domain, seed, params) for seed in seeds]
    with Pool(processes=TEST_WORKERS) as pool:
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


def save_cell_result(tuned_on, tested_on, times, n_optimal):
    results = load_results()
    results.setdefault(tuned_on, {})[tested_on] = {"times": times, "n_optimal": n_optimal}
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


if __name__ == "__main__":
    row_labels = ["ba", "rr", "ws", "mixed", "default"]
    test_domains = ["rr", "ws"]  # BA is NOT re-evaluated -- reused from the original results file

    tuned_params = {}
    for d in row_labels:
        path = os.path.join(SG, f"rpb_tuned_{d}.json")
        with open(path) as f:
            tuned_params[d] = json.load(f)
        print(f"RPB-tuned-on-{d} params: {tuned_params[d]}", flush=True)

    existing = load_results()
    print(f"\n=== Recalibrated RR/WS cross-test (RR: n={RR_N_NODES},degree={RR_DEGREE} | "
          f"WS: n={WS_N_NODES},k={WS_K},p={WS_P}) ===", flush=True)
    for tuned_on in row_labels:
        for tested_on in test_domains:
            if tuned_on in existing and tested_on in existing[tuned_on]:
                times = existing[tuned_on][tested_on]["times"]
                n_optimal = existing[tuned_on][tested_on]["n_optimal"]
                print(f"RPB-tuned-on-{tuned_on} on {tested_on}: [resumed] "
                      f"geomean={geometric_mean_stable(times):.2f}s optimal={n_optimal}/{len(times)}", flush=True)
                continue
            times, n_optimal = evaluate_cell(tuned_on, tested_on, tuned_params[tuned_on])
            geo = geometric_mean_stable(times)
            save_cell_result(tuned_on, tested_on, times, n_optimal)
            print(f"RPB-tuned-on-{tuned_on} on {tested_on}: geomean={geo:.2f}s "
                  f"optimal={n_optimal}/{len(times)}", flush=True)

    # Merge with BA's existing (unchanged) results and print the final combined table
    with open(os.path.join(SG, "crosstest_rpb_results.json")) as f:
        original = json.load(f)
    new_results = load_results()

    print("\n=== Final combined table (BA reused from original run1, RR/WS recalibrated) ===", flush=True)
    pooled_rows = {}
    for row in row_labels:
        ba_times = original[row]["ba"]["times"]
        rr_times = new_results[row]["rr"]["times"]
        ws_times = new_results[row]["ws"]["times"]
        pooled = ba_times + rr_times + ws_times
        pooled_rows[row] = pooled
        print(f"{row}: ba={geometric_mean_stable(ba_times):.2f} rr={geometric_mean_stable(rr_times):.2f} "
              f"ws={geometric_mean_stable(ws_times):.2f} pooled={geometric_mean_stable(pooled):.2f}", flush=True)

    oracle = [min(pooled_rows[r][i] for r in row_labels) for i in range(90)]
    print(f"Oracle (all 5): pooled_geomean={geometric_mean_stable(oracle):.2f}s", flush=True)

    win_counts = {r: 0 for r in row_labels}
    for i in range(90):
        best = min(row_labels, key=lambda r: pooled_rows[r][i])
        win_counts[best] += 1
    print(f"Win counts across 90 instances: {win_counts}", flush=True)
    print("CROSSTEST_RPB_RECALIBRATED_DONE", flush=True)
