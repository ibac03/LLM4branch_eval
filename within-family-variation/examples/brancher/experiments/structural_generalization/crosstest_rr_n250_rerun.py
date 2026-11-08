"""
Re-run of Original / Tuned-via-BA / Tuned-via-ER / Tuned-via-RR on RR,
now at the CORRECT n=250/degree=3 tier (the original crosstest_rr_phase4.py
hardcoded n_nodes=200, before the n=250 recalibration) -- same 30 seeds
(1000-1029) as before, so all 5 rows of the final table are directly
comparable at last.
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

SG = os.path.dirname(__file__)
PROGRAMS = {
    "Original": os.path.join(SG, "..", "..", "program", "indset", "program.py"),
    "Tuned-via-BA": os.path.join(SG, "program_category_k_ba_tuned.py"),
    "Tuned-via-ER": os.path.join(SG, "program_category_k_tuned.py"),
    "Tuned-via-RR": os.path.join(SG, "program_category_rr_tuned.py"),
}
RR_SEEDS = list(range(1000, 1030))


def solve_one(args):
    label, seed = args
    score_function, used_features, function_params, _ = load_program(PROGRAMS[label])
    score_function = partial(score_function, params=function_params)
    instance = generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


if __name__ == "__main__":
    for label in PROGRAMS:
        print(f"=== {label} on RR (n=250, degree=3) starting ===", flush=True)
        tasks = [(label, seed) for seed in RR_SEEDS]
        with Pool(processes=4) as pool:
            results = pool.map(solve_one, tasks)
        times = []
        for seed, t, n, s in sorted(results):
            print(f"seed={seed}: {label}(time={t:.2f}, nodes={n}, status={s})", flush=True)
            times.append(t)
        print(f"{label}-on-RR250 geomean: {geometric_mean_stable(times):.2f}", flush=True)
        print(f"{label}-on-RR250 avg_time: {np.mean(times):.2f}", flush=True)
        print(f"=== {label} done ===", flush=True)
    print("RR_N250_RERUN_DONE", flush=True)
