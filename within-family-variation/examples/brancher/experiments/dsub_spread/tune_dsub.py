"""
Re-run LLM4Branch's Bayesian-optimization parameter-tuning stage (faithfully
matching evaluator.py's params_optimizer: skopt, acq_func='gp_hedge',
random_state=123, time_limit=180 for tuning solves, target='nodes') at
different D_sub sizes (4 = paper's actual setting, 40, 100), to test whether
per-instance performance spread on held-out instances shrinks as D_sub grows.

D_sub sets are nested: D_sub=4 uses seeds [666666..666669], D_sub=40 uses
[666666..666705] (the same 4 plus 36 more), D_sub=100 uses [666666..666765]
(the same 40 plus 60 more) -- isolating dataset size as the only variable.

Deviates from evaluator.py in exactly one place, deliberately: instead of
seeding the search with the *current* (already D_sub=4-tuned) PARAMS from
program.py -- which would bias every D_sub arm toward that specific solution
-- all n_initial points are drawn as skopt's own random initial batch, with
no manually-injected first point. This keeps all three D_sub runs starting
from the same neutral, reproducible (fixed seed) exploration, so any
difference in outcome is attributable to D_sub size itself.

Usage:
    python tune_dsub.py --d_sub 4
    python tune_dsub.py --d_sub 40
    python tune_dsub.py --d_sub 100
"""
import argparse
import multiprocessing
import os
import sys

from multiprocessing import Pool
from skopt import Optimizer
from skopt.space import Real

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import numerical_mean_stable, get_scip_params, load_program
from evaluator import execute_ecole

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "setcover", "program.py")
DATASET = "setcover"
TRAIN_SEED_BASE = 666666  # matches LLM4Branch's own train_seed, nested ranges from here
FAILURE_PENALTY = 1e9      # evaluator.py returns 0.0 on failure, which would look
                           # like the *best possible* score under minimization --
                           # a bug we deliberately don't replicate.


def objective(function_params, seeds, score_function, used_features, scip_params, pool, cores_list, lock_dir):
    tasks = [
        ('evolve', DATASET, seed, cores_list, lock_dir, score_function, used_features, function_params, scip_params, None, None, None)
        for seed in seeds
    ]
    results = pool.map(execute_ecole, tasks)
    for r in results:
        if not r["validity"]:
            print(f"  FAILED seed={r.get('seed')}: {r.get('error_info')}", flush=True)
            return FAILURE_PENALTY
    mean_nodes = numerical_mean_stable([r["nodes"] for r in results])
    print(f"  params={[round(p, 4) for p in function_params]} -> mean_nodes={mean_nodes:.2f}", flush=True)
    return mean_nodes


def tune(d_sub, max_workers, cores_list, lock_dir, n_calls, n_initial, random_state=123):
    score_function, used_features, _unused_params, bound_params = load_program(PROGRAM_PATH)
    scip_params = get_scip_params(time_limit=180)
    seeds = list(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + d_sub))

    param_space = [Real(lo, hi) for lo, hi in bound_params]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=random_state)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        x0 = opt.ask(n_initial)
        y0 = [objective(x, seeds, score_function, used_features, scip_params, pool, cores_list, lock_dir) for x in x0]
        opt.tell(x0, y0)
        for i in range(n_initial, n_calls):
            print(f"D_sub={d_sub}: iteration {i + 1}/{n_calls}", flush=True)
            next_x = opt.ask()
            f_val = objective(next_x, seeds, score_function, used_features, scip_params, pool, cores_list, lock_dir)
            opt.tell(next_x, f_val)

    result = opt.get_result()
    return result.x, result.fun, used_features, bound_params


def write_program(theta, used_features, bound_params, out_path):
    with open(PROGRAM_PATH) as f:
        original_source = f.read()
    body = original_source.split("\n", 3)[3]  # drop the USED_FEATURES/PARAMS/BOUNDS header lines
    with open(out_path, "w") as f:
        f.write(f"USED_FEATURES = {used_features}\n")
        f.write(f"PARAMS = {list(theta)}\n")
        f.write(f"BOUNDS = {bound_params}\n")
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_sub", type=int, required=True)
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--cores_list", default="0-5")
    parser.add_argument("--n_calls", type=int, default=50, help="total BO evaluations (paper: 50)")
    parser.add_argument("--n_initial", type=int, default=10, help="random warm-up evaluations before acquisition-guided search")
    parser.add_argument("--random_state", type=int, default=123, help="skopt Optimizer random_state (varies for ZO seed-variance experiment)")
    parser.add_argument("--out_suffix", default="", help="extra suffix for the output filename, e.g. a seed tag")
    args = parser.parse_args()

    start, end = args.cores_list.split("-")
    cores_list = list(range(int(start), int(end) + 1))
    lock_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "cpu_locks", f"dsub_{args.d_sub}{args.out_suffix}")
    os.makedirs(lock_dir, exist_ok=True)

    print(f"Tuning with D_sub={args.d_sub} (seeds {TRAIN_SEED_BASE}..{TRAIN_SEED_BASE + args.d_sub - 1}), random_state={args.random_state}")
    theta, fun, used_features, bound_params = tune(args.d_sub, args.max_workers, cores_list, lock_dir, args.n_calls, args.n_initial, args.random_state)
    print(f"\nD_sub={args.d_sub} DONE: best mean_nodes={fun:.2f}, theta={theta}")

    out_path = os.path.join(os.path.dirname(__file__), f"program_dsub{args.d_sub}{args.out_suffix}.py")
    write_program(theta, used_features, bound_params, out_path)
    print(f"Saved: {out_path}")
