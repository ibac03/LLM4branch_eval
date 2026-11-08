"""
Recalibrate RR test set to target ~20s geomean (up from the previous
~1.8-2.6s at n=200/degree=3). Sweep n at degree=3 (known-good degree
from Phase 1) since n=200->2.58s and n=400 timed out completely --
need an intermediate size. Always check status explicitly.
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
from category_rr_indset import generate_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")


def solve_one(args):
    seed, degree, n_nodes = args
    score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
    score_function = partial(score_function, params=function_params)
    instance = generate_instance(seed=seed, n_nodes=n_nodes, degree=degree)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return t, n, s


if __name__ == "__main__":
    print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)
    for n_nodes in [220, 250, 280, 300, 330]:
        with Pool(processes=4) as pool:
            tasks = [(seed, 3, n_nodes) for seed in range(4)]
            results = pool.map(solve_one, tasks)
        times = [r[0] for r in results]
        nodes_list = [r[1] for r in results]
        statuses = [r[2] for r in results]
        all_optimal = all(s == "optimal" for s in statuses)
        print(f"RR n={n_nodes}, degree=3: nodes={nodes_list} time={[round(t,2) for t in times]} "
              f"statuses={statuses} all_optimal={all_optimal} "
              f"geomean_time={geometric_mean_stable(times):.2f} mean_nodes={np.mean(nodes_list):.1f}", flush=True)
