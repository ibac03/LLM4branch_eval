"""
regular.md Phase 2: cross-test Original / Tuned-via-ER / Tuned-via-BA
on held-out RR (n=200, degree=3, calibrated Phase 1 regime), 30 fresh
seeds (1000-1029, disjoint from calibration seeds 0-3).
"""
import sys
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch
from category_rr_indset import generate_instance

SG = os.path.dirname(__file__)
PROGRAMS = {
    "Original": os.path.join(SG, "..", "..", "program", "indset", "program.py"),
    "Tuned-via-ER": os.path.join(SG, "program_category_k_tuned.py"),
    "Tuned-via-BA": os.path.join(SG, "program_category_k_ba_tuned.py"),
}
N_NODES = 200
DEGREE = 3
SEEDS = list(range(1000, 1030))


def solve_one(args):
    seed, program_path = args
    score_function, used_features, function_params, _ = load_program(program_path)
    score_function = partial(score_function, params=function_params)
    instance = generate_instance(seed=seed, n_nodes=N_NODES, degree=DEGREE)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


if __name__ == "__main__":
    for label, program_path in PROGRAMS.items():
        print(f"=== RR (n={N_NODES}, degree={DEGREE}): {label} starting ===", flush=True)
        tasks = [(seed, program_path) for seed in SEEDS]
        with Pool(processes=4) as pool:
            results = pool.map(solve_one, tasks)
        times = []
        for seed, t, n, s in sorted(results):
            print(f"seed={seed}: {label}(time={t:.2f}, nodes={n}, status={s})", flush=True)
            times.append(t)
        print(f"{label}-on-RR geomean: {geometric_mean_stable(times):.2f}", flush=True)
        print(f"{label}-on-RR avg_time: {np.mean(times):.2f}", flush=True)
        print(f"=== {label} done ===", flush=True)
    print("RR_PHASE2_DONE", flush=True)
