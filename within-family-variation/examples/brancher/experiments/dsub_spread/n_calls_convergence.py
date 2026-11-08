"""
Does the BO tuning procedure's instability come from too little search budget
(n_calls=50 isn't enough to explore the 7-D parameter space), or from genuine
multi-modality (different seeds converge to different, individually-stable
local optima)? Extends ONE continuous tuning trajectory (fixed D_sub=4, fixed
random_state) out to 200 calls, checkpointing the running-best score at
50/100/150/200 -- if the score is still improving substantially past 50,
that's under-exploration; if it plateaus well before 50, that's evidence of
a stable local optimum reached quickly (pointing toward multi-modality as the
real explanation for the seed-to-seed dispersion found separately).

Uses the same random_state=1 as the first ZO seed-variance run, so this
run's own 50-call checkpoint should closely reproduce that existing result.

Usage:
    python n_calls_convergence.py
"""
import argparse
import os
import sys

from multiprocessing import get_context
from skopt import Optimizer
from skopt.space import Real

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, load_program
from tune_dsub import objective, write_program, PROGRAM_PATH, TRAIN_SEED_BASE

CHECKPOINTS = [50, 100, 150, 200]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_sub", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--cores_list", default="0-5")
    parser.add_argument("--random_state", type=int, default=1)
    args = parser.parse_args()

    start, end = args.cores_list.split("-")
    cores_list = list(range(int(start), int(end) + 1))
    lock_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "cpu_locks", "n_calls_convergence")
    os.makedirs(lock_dir, exist_ok=True)

    score_function, used_features, _unused_params, bound_params = load_program(PROGRAM_PATH)
    scip_params = get_scip_params(time_limit=180)
    seeds = list(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + args.d_sub))

    param_space = [Real(lo, hi) for lo, hi in bound_params]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=args.random_state)

    ctx = get_context("fork")
    n_initial = 10
    max_calls = max(CHECKPOINTS)

    with ctx.Pool(processes=args.max_workers) as pool:
        x0 = opt.ask(n_initial)
        y0 = [objective(x, seeds, score_function, used_features, scip_params, pool, cores_list, lock_dir) for x in x0]
        opt.tell(x0, y0)

        for i in range(n_initial, max_calls):
            call_num = i + 1
            next_x = opt.ask()
            f_val = objective(next_x, seeds, score_function, used_features, scip_params, pool, cores_list, lock_dir)
            opt.tell(next_x, f_val)

            if call_num in CHECKPOINTS:
                result = opt.get_result()
                print(f"\n=== CHECKPOINT n_calls={call_num}: best mean_nodes={result.fun:.2f}, theta={list(result.x)} ===\n", flush=True)
                out_path = os.path.join(os.path.dirname(__file__), f"program_ncalls{call_num}.py")
                write_program(result.x, used_features, bound_params, out_path)
                print(f"Saved: {out_path}", flush=True)
            else:
                print(f"D_sub={args.d_sub} random_state={args.random_state}: call {call_num}/{max_calls}", flush=True)

    print("\nDone.")
