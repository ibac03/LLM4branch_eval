"""
Calibrate Watts-Strogatz parameters (n, k, beta) before committing to
LLM evolution compute. Looking for a regime that is:
  1. sufficiently difficult for SCIP (nontrivial, not pathological/all
     timeouts);
  2. leaves substantial room for improvement for BOTH existing policies
     (P_BA=Original, P_RR=Evolved-RR-tuned-on-BA) -- i.e. neither one
     already solves it trivially fast, which would mean WS doesn't add
     a genuinely new challenge.

Uses calibration seeds 100-103 -- separate from seeds 0-3 (used for any
future WS-specific training) and separate from whatever range becomes
the eventual 30 held-out WS test seeds.
"""
import sys
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch, TIME_LIMIT
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
PROGRAMS = {
    "Original": os.path.join(SG, "..", "..", "program", "indset", "program.py"),
    "Evolved-RR-tuned-on-BA": os.path.join(SG, "program_evolved_rr_ba_tuned.py"),
}
CALIB_SEEDS = [100, 101, 102, 103]


def solve_one(args):
    label, seed, n_nodes, k, p = args
    score_function, used_features, function_params, _ = load_program(PROGRAMS[label])
    score_function = partial(score_function, params=function_params)
    instance = generate_ws_instance(seed=seed, n_nodes=n_nodes, k=k, p=p)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return t, n, s


if __name__ == "__main__":
    print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)
    configs = [
        (200, 4, 0.01), (200, 4, 0.05), (200, 4, 0.1), (200, 4, 0.2),
        (200, 6, 0.05), (200, 6, 0.1),
    ]
    for n_nodes, k, p in configs:
        for label in PROGRAMS:
            with Pool(processes=4) as pool:
                tasks = [(label, seed, n_nodes, k, p) for seed in CALIB_SEEDS]
                results = pool.map(solve_one, tasks)
            times = [r[0] for r in results]
            nodes_list = [r[1] for r in results]
            statuses = [r[2] for r in results]
            all_optimal = all(s == "optimal" for s in statuses)
            print(f"WS n={n_nodes} k={k} p={p} | {label}: nodes={nodes_list} "
                  f"time={[round(t,2) for t in times]} statuses={statuses} "
                  f"all_optimal={all_optimal} geomean_time={geometric_mean_stable(times):.2f} "
                  f"mean_nodes={np.mean(nodes_list):.1f}", flush=True)
    print("WS_CALIBRATION_DONE", flush=True)
