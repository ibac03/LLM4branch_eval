"""
regular.md Phase 4: crossover crosstest. Evaluate Tuned-via-RR
(program_category_rr_tuned.py) on all three domains -- BA-standard,
ER-large, and RR -- using the exact same test seeds already used for
Original/Tuned-via-ER/Tuned-via-BA, so the final crossover table is
directly comparable across all 4 policies x 3 domains.
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
from category_rr_indset import generate_instance as generate_rr_instance
from category_k_er_indset_large import generate_instance as generate_er_large_instance

SG = os.path.dirname(__file__)
PROGRAM = os.path.join(SG, "program_category_rr_tuned.py")

RR_SEEDS = list(range(1000, 1030))
ER_SEEDS = list(range(600, 630))


def solve_rr(seed):
    score_function, used_features, function_params, _ = load_program(PROGRAM)
    score_function = partial(score_function, params=function_params)
    instance = generate_rr_instance(seed=seed, n_nodes=200, degree=3)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


def solve_er(seed):
    score_function, used_features, function_params, _ = load_program(PROGRAM)
    score_function = partial(score_function, params=function_params)
    instance = generate_er_large_instance(seed=seed)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


if __name__ == "__main__":
    print("=== Tuned-via-RR on RR (n=200, degree=3) starting ===", flush=True)
    with Pool(processes=4) as pool:
        results = pool.map(solve_rr, RR_SEEDS)
    times = []
    for seed, t, n, s in sorted(results):
        print(f"seed={seed}: Tuned-via-RR-on-RR(time={t:.2f}, nodes={n}, status={s})", flush=True)
        times.append(t)
    print(f"Tuned-via-RR-on-RR geomean: {geometric_mean_stable(times):.2f}", flush=True)
    print(f"Tuned-via-RR-on-RR avg_time: {np.mean(times):.2f}", flush=True)
    print("=== Tuned-via-RR on RR done ===", flush=True)

    print("=== Tuned-via-RR on ER-large (n=1000, p=0.0036) starting ===", flush=True)
    with Pool(processes=4) as pool:
        results = pool.map(solve_er, ER_SEEDS)
    times = []
    for seed, t, n, s in sorted(results):
        print(f"seed={seed}: Tuned-via-RR-on-ER(time={t:.2f}, nodes={n}, status={s})", flush=True)
        times.append(t)
    print(f"Tuned-via-RR-on-ER geomean: {geometric_mean_stable(times):.2f}", flush=True)
    print(f"Tuned-via-RR-on-ER avg_time: {np.mean(times):.2f}", flush=True)
    print("=== Tuned-via-RR on ER-large done ===", flush=True)

    print("RR_PHASE4_ER_RR_DONE", flush=True)
