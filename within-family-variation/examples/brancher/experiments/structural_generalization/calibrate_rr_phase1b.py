"""
regular.md Phase 1 (revised): the original n=1000 sweep was too slow
single-threaded (degree=4's first seed alone exceeded 120s). Switching
to n=400 (matching Category K's ER training tier size, already known
tractable) and parallelizing across seeds with a worker pool.
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
    N_NODES = 400
    with Pool(processes=4) as pool:
        for degree in [3, 4, 6, 8, 10, 12]:
            tasks = [(seed, degree, N_NODES) for seed in range(4)]
            results = pool.map(solve_one, tasks)
            times = [r[0] for r in results]
            nodes_list = [r[1] for r in results]
            statuses = [r[2] for r in results]
            all_optimal = all(s == "optimal" for s in statuses)
            print(f"RR n={N_NODES}, degree={degree}: nodes={nodes_list} time={[round(t,2) for t in times]} "
                  f"statuses={statuses} all_optimal={all_optimal} "
                  f"geomean_time={geometric_mean_stable(times):.2f} mean_nodes={np.mean(nodes_list):.1f}", flush=True)
