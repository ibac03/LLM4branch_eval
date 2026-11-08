"""
regular.md Phase 1: find a hard-but-tractable random-regular degree
regime at n=1000, using the Original released IndSet policy only.
Always checks status explicitly (no repeat of the earlier
timeout-mistaken-for-difficulty mistake).
"""
import sys
import os
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch, TIME_LIMIT
from category_rr_indset import generate_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
score_function = partial(score_function, params=function_params)

print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)

for degree in [4, 6, 8, 10, 12]:
    times, nodes_list, statuses = [], [], []
    for seed in range(4):
        instance = generate_instance(seed=seed, n_nodes=1000, degree=degree)
        t, n, s = solve_llm4branch(instance, score_function, used_features)
        times.append(t)
        nodes_list.append(n)
        statuses.append(s)
    all_optimal = all(s == "optimal" for s in statuses)
    print(f"RR n=1000, degree={degree}: nodes={nodes_list} time={[round(t,2) for t in times]} "
          f"statuses={statuses} all_optimal={all_optimal} "
          f"geomean_time={geometric_mean_stable(times):.2f} mean_nodes={np.mean(nodes_list):.1f}", flush=True)
