"""
Evaluate program_evolved_rr_ba_tuned.py (the evolved RR structure,
params re-tuned for BA) on RR(n=250, seeds 1000-1029) -- the reverse
specialization check: did BA-tuning hurt its RR performance?
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
PROGRAM = os.path.join(SG, "program_evolved_rr_ba_tuned.py")
RR_SEEDS = list(range(1000, 1030))


def solve_rr(seed):
    score_function, used_features, function_params, _ = load_program(PROGRAM)
    score_function = partial(score_function, params=function_params)
    instance = generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


if __name__ == "__main__":
    print("=== Evolved-RR-tuned-on-BA on RR (n=250, degree=3) starting ===", flush=True)
    with Pool(processes=4) as pool:
        results = pool.map(solve_rr, RR_SEEDS)
    times = []
    for seed, t, n, s in sorted(results):
        print(f"seed={seed}: Evolved-RR-tuned-on-BA-on-RR(time={t:.2f}, nodes={n}, status={s})", flush=True)
        times.append(t)
    print(f"Evolved-RR-tuned-on-BA-on-RR geomean: {geometric_mean_stable(times):.2f}", flush=True)
    print(f"Evolved-RR-tuned-on-BA-on-RR avg_time: {np.mean(times):.2f}", flush=True)
    print("EVOLVED_RR_BA_TUNED_ON_RR_DONE", flush=True)
