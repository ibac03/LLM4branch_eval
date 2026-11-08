"""
Full 30-instance RR test-set evaluation for Evolved-WS (native params) --
screen_evolved_ws.py confirmed native params pass on RR (0.95x vs RPB,
all-optimal on 4 screen seeds), so no retune needed here. Same seeds
(1000-1029, n=250, degree=3) already used for Original/Evolved-RR on RR.
"""
import sys
import os
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch, TIME_LIMIT
from category_rr_indset import generate_instance as generate_rr_instance

SG = os.path.dirname(__file__)
PROGRAM = os.path.join(SG, "ws_evolution_output", "best", "best_program.py")
RR_SEEDS = list(range(1000, 1030))


def solve_one(seed):
    score_function, used_features, function_params, _ = load_program(PROGRAM)
    score_function = partial(score_function, params=function_params)
    instance = generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


if __name__ == "__main__":
    print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)
    print("=== Evolved-WS on RR (n=250, degree=3) starting ===", flush=True)
    with Pool(processes=4) as pool:
        results = pool.map(solve_one, RR_SEEDS)
    results.sort()
    times = []
    for seed, t, n, s in results:
        print(f"seed={seed}: Evolved-WS-on-RR(time={t:.2f}, nodes={n}, status={s})", flush=True)
        times.append(t)
    geo = geometric_mean_stable(times)
    print(f"Evolved-WS-on-RR geomean: {geo:.2f}", flush=True)
    print("EVAL_EVOLVED_WS_ON_RR_DONE", flush=True)
