"""
WS calibration round 4: found a promising regime at n=200,k=20,p=0.15
(Original geomean=1.34s, Evolved-RR-tuned-on-BA geomean=1.38s, both
comparable, real branching ~600 nodes). Refining around this point
toward a harder target (~20-30s, matching RR's calibrated difficulty).
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
        (200, 20, 0.2), (200, 20, 0.25), (200, 20, 0.3),
        (250, 20, 0.15), (300, 20, 0.15),
        (200, 24, 0.15), (200, 24, 0.2),
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
    print("WS_CALIBRATION4_DONE", flush=True)
