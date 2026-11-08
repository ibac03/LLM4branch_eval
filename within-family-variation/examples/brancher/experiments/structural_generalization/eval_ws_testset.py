"""
Final WS held-out test-set evaluation for the oracle table: Original
(native), Evolved-RR (native), Evolved-WS (native) on 30 fresh WS
instances (n=200,k=20,p=0.25 -- the calibrated test tier), seeds
200-229 (distinct from calibration seeds 100-103).

Per the established oracle-table methodology: native params are used for
all three here (WS screen for Original/Evolved-RR already confirmed no
retune needed; Evolved-WS is evaluated natively on its own home domain
by construction).
"""
import sys
import os
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch, TIME_LIMIT
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
PROGRAMS = {
    "Original": os.path.join(SG, "..", "..", "program", "indset", "program.py"),
    "Evolved-RR": os.path.join(SG, "rr_evolution_output", "best", "best_program.py"),
    "Evolved-WS": os.path.join(SG, "ws_evolution_output", "best", "best_program.py"),
}
WS_TEST_SEEDS = list(range(200, 230))


def solve_one(args):
    label, seed = args
    score_function, used_features, function_params, _ = load_program(PROGRAMS[label])
    score_function = partial(score_function, params=function_params)
    instance = generate_ws_instance(seed=seed)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return seed, t, n, s


if __name__ == "__main__":
    print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)
    print(f"WS_TEST_SEEDS={WS_TEST_SEEDS}", flush=True)

    for label in PROGRAMS:
        print(f"=== {label} on WS (n=200,k=20,p=0.25) starting ===", flush=True)
        with Pool(processes=4) as pool:
            tasks = [(label, seed) for seed in WS_TEST_SEEDS]
            results = pool.map(solve_one, tasks)
        results.sort()
        times = []
        for seed, t, n, s in results:
            print(f"seed={seed}: {label}(time={t:.2f}, nodes={n}, status={s})", flush=True)
            times.append(t)
        geo = geometric_mean_stable(times)
        print(f"{label}-on-WS geomean: {geo:.2f}", flush=True)
        print(f"=== {label} on WS done ===", flush=True)

    print("EVAL_WS_TESTSET_DONE", flush=True)
