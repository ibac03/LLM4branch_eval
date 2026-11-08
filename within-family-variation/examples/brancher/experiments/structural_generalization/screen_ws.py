"""
Cheap screen: RPB vs Original (native) vs Evolved-RR (native, untouched
RR-trained params) on a small WS sample (n=200,k=20,p=0.25). Decides
whether either needs a WS-specific retune before we build the final
table, and confirms WS is a genuinely fresh challenge worth its own
evolved structure.
"""
import sys
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch, solve_rpb, TIME_LIMIT
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
PROGRAMS = {
    "Original (native)": os.path.join(SG, "..", "..", "program", "indset", "program.py"),
    "Evolved-RR (native)": os.path.join(SG, "rr_evolution_output", "best", "best_program.py"),
}
CALIB_SEEDS = [100, 101, 102, 103]


def solve_policy(args):
    label, seed = args
    score_function, used_features, function_params, _ = load_program(PROGRAMS[label])
    score_function = partial(score_function, params=function_params)
    instance = generate_ws_instance(seed=seed)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return t, n, s


def solve_rpb_one(seed):
    instance = generate_ws_instance(seed=seed)
    t, n, s = solve_rpb(instance)
    return t, n, s


if __name__ == "__main__":
    print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)

    print("=== RPB on WS ===", flush=True)
    with Pool(processes=4) as pool:
        rpb_results = pool.map(solve_rpb_one, CALIB_SEEDS)
    rpb_times = [r[0] for r in rpb_results]
    rpb_geo = geometric_mean_stable(rpb_times)
    print(f"RPB: times={[round(t,2) for t in rpb_times]} statuses={[r[2] for r in rpb_results]} "
          f"geomean={rpb_geo:.2f}", flush=True)

    for label in PROGRAMS:
        with Pool(processes=4) as pool:
            tasks = [(label, seed) for seed in CALIB_SEEDS]
            results = pool.map(solve_policy, tasks)
        times = [r[0] for r in results]
        statuses = [r[2] for r in results]
        all_optimal = all(s == "optimal" for s in statuses)
        geo = geometric_mean_stable(times)
        speedup = rpb_geo / geo if geo > 0 else 0
        print(f"{label}: times={[round(t,2) for t in times]} statuses={statuses} "
              f"all_optimal={all_optimal} geomean={geo:.2f} speedup_vs_RPB={speedup:.2f}x", flush=True)

    print("WS_SCREEN_DONE", flush=True)
