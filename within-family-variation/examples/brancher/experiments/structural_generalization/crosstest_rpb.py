"""
Full 3x3 cross-test: RPB (relpscost) tuned per domain (via tune_rpb.py) x
tested on each domain's held-out test set -- same exact seeds/tiers/time
caps as every other row of the oracle table.

BA: n_nodes=1000,affinity=4 (medium), seeds 999999999..1000000028, 3600s cap
RR: n_nodes=250,degree=3, seeds 1000..1029, 120s cap
WS: n_nodes=200,k=20,p=0.25, seeds 200..229, 120s cap
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, geometric_mean_stable, create_instance
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance
from tune_rpb import PARAM_SPACE

BA_SEEDS = list(range(999999999, 999999999 + 30))
RR_SEEDS = list(range(1000, 1030))
WS_SEEDS = list(range(200, 230))
TEST_SEEDS = {"ba": BA_SEEDS, "rr": RR_SEEDS, "ws": WS_SEEDS}
TEST_TIME_LIMIT = {"ba": 3600, "rr": 120, "ws": 120}
TEST_WORKERS = {"ba": 3, "rr": 6, "ws": 6}

SG = os.path.dirname(__file__)


def make_test_instance(domain, seed):
    if domain == "ba":
        return create_instance("indset", seed, "medium")
    elif domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    else:
        return generate_ws_instance(seed=seed, n_nodes=200, k=20, p=0.25)


def solve_one(args):
    import ecole
    tuned_domain, test_domain, seed, params = args
    instance = make_test_instance(test_domain, seed)
    scip_params = {
        **get_scip_params(time_limit=TEST_TIME_LIMIT[test_domain]),
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
    with Pool(processes=TEST_WORKERS[test_domain]) as pool:
        results = pool.map(solve_one, tasks)
    results.sort()
    times = [t for _, t, _, _ in results]
    n_optimal = sum(1 for *_, s in results if s == "optimal")
    return times, n_optimal


RESULTS_PATH = os.path.join(SG, "crosstest_rpb_results.json")  # overwritten in __main__ if --run_suffix given


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_cell_result(tuned_on, tested_on, times, n_optimal):
    # Resumable, incremental save: written after EVERY cell completes, not
    # just at the end -- a killed/crashed run leaves a valid partial
    # results file instead of only scattered log lines, and a rerun can
    # skip cells already present here (see the `if key in results: skip`
    # check in the main loop below).
    results = load_results()
    results.setdefault(tuned_on, {})[tested_on] = {"times": times, "n_optimal": n_optimal}
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)  # atomic, avoids a torn/partial file if killed mid-write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_suffix", default="",
                         help="e.g. '_run2' -- loads rpb_tuned_{domain}{suffix}.json for the "
                              "BO-tuned rows (ba/rr/ws/mixed, from a differently-seeded tune_rpb.py "
                              "run) and writes/resumes crosstest_rpb_results{suffix}.json, so repeat "
                              "runs (to check tuning-variance / measurement noise) don't clobber the "
                              "original results. 'default' has no tuning step, so it always reads "
                              "the plain rpb_tuned_default.json regardless of suffix -- but IS still "
                              "re-evaluated fresh under each suffix (no params vary, so this "
                              "specifically measures pure run-to-run timing/measurement noise, not "
                              "tuning variance, as a noise-floor reference for the tuned rows).")
    args = parser.parse_args()
    RESULTS_PATH = os.path.join(SG, f"crosstest_rpb_results{args.run_suffix}.json")

    domains = ["ba", "rr", "ws"]  # tested-on columns: always the 3 real structural regimes
    # tuned-on rows: the 3 domain-specific tunings, plus "mixed" (BO-tuned on
    # a pooled 3-seeds-per-domain training set, see tune_rpb.py's "mixed"
    # domain) and "default" (SCIP's own untuned relpscost, params={}) --
    # both added as extra reference rows on the same 3 test columns, per the
    # user's request to see where domain-specific tuning sits relative to
    # both a completely generic tuning and no tuning at all.
    row_labels = ["ba", "rr", "ws", "mixed", "default"]
    tuned_params = {}
    for d in row_labels:
        suffix = args.run_suffix if d != "default" else ""
        path = os.path.join(SG, f"rpb_tuned_{d}{suffix}.json")
        with open(path) as f:
            tuned_params[d] = json.load(f)
        print(f"RPB-tuned-on-{d} params: {tuned_params[d]}", flush=True)

    existing = load_results()
    cell_times = {tuned: {} for tuned in row_labels}
    print("\n=== Cross-test (rows=RPB tuned on, cols=tested on) ===", flush=True)
    for tuned_on in row_labels:
        for tested_on in domains:
            if tuned_on in existing and tested_on in existing[tuned_on]:
                times = existing[tuned_on][tested_on]["times"]
                n_optimal = existing[tuned_on][tested_on]["n_optimal"]
                print(f"RPB-tuned-on-{tuned_on} on {tested_on}: [resumed from {RESULTS_PATH}] "
                      f"geomean={geometric_mean_stable(times):.2f}s optimal={n_optimal}/{len(times)}",
                      flush=True)
                cell_times[tuned_on][tested_on] = times
                continue
            times, n_optimal = evaluate_cell(tuned_on, tested_on, tuned_params[tuned_on])
            geo = geometric_mean_stable(times)
            cell_times[tuned_on][tested_on] = times
            save_cell_result(tuned_on, tested_on, times, n_optimal)
            print(f"RPB-tuned-on-{tuned_on} on {tested_on}: geomean={geo:.2f}s "
                  f"optimal={n_optimal}/{len(times)}", flush=True)

    print("\n=== Per-tuning pooled 90-instance geomean (BA+RR+WS test sets) ===", flush=True)
    pooled_rows = {}
    for tuned_on in row_labels:
        pooled = cell_times[tuned_on]["ba"] + cell_times[tuned_on]["rr"] + cell_times[tuned_on]["ws"]
        pooled_rows[tuned_on] = pooled
        print(f"RPB-tuned-on-{tuned_on}: pooled_geomean={geometric_mean_stable(pooled):.2f}s", flush=True)

    oracle_times = [min(pooled_rows[d][i] for d in row_labels) for i in range(90)]
    print(f"Oracle (per-instance min across all {len(row_labels)} RPB tunings): "
          f"pooled_geomean={geometric_mean_stable(oracle_times):.2f}s", flush=True)

    domain_only_oracle_times = [min(pooled_rows["ba"][i], pooled_rows["rr"][i], pooled_rows["ws"][i]) for i in range(90)]
    print(f"Oracle (per-instance min across only the 3 domain-specific RPB tunings): "
          f"pooled_geomean={geometric_mean_stable(domain_only_oracle_times):.2f}s", flush=True)

    win_counts = {d: 0 for d in row_labels}
    for i in range(90):
        best = min(row_labels, key=lambda d: pooled_rows[d][i])
        win_counts[best] += 1
    print(f"Win counts across 90 instances: {win_counts}", flush=True)
    print("CROSSTEST_RPB_DONE", flush=True)
