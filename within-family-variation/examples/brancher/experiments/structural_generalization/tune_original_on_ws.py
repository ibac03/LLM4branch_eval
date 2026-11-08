"""
Cheap first step before committing to LLM evolution for WS: BO-tune
Original's own 4-feature structure (obj_coeff, reduced_cost,
fractionality, constraint_count) specifically for WS
(n=200, k=20, p=0.25 -- calibrated, all-optimal, ~20-30s geomean).

If this alone gets competitive WS performance, we may not need a new
evolved structure at all for this category.
"""
import argparse
import multiprocessing
import os
import sys

import ecole
import numpy as np
from skopt import Optimizer
from skopt.space import Real

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import numerical_mean_stable, get_scip_params, load_program, normalize
from category_ws_indset import generate_instance as generate_ws_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
TRAIN_SEEDS = [0, 1, 2, 3]
FAILURE_PENALTY = 1e9


def solve_one(args):
    seed, used_features, function_params, scip_params = args
    from functools import partial as _partial
    score_function, _uf, _fp, _bp = load_program(PROGRAM_PATH)
    score_function = _partial(score_function, params=function_params)

    instance = generate_ws_instance(seed=seed)

    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=scip_params,
    )
    env.seed(0)
    try:
        observation, action_set, _, done, info = env.reset(instance)
        while not done:
            variable_features = np.concatenate(
                (observation["Node"].variable_features[action_set, :],
                 observation["Khalil"].features[action_set, :]), axis=1)
            variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
            variable_features = normalize(variable_features, used_features)
            scores = score_function(variable_features)
            action = action_set[scores.argmax()]
            observation, action_set, _, done, info = env.step(action)
        m = env.model.as_pyscipopt()
        return {"validity": True, "seed": seed, "nodes": m.getNNodes()}
    except Exception as e:
        return {"validity": False, "seed": seed, "error_info": str(e)}


def objective(function_params, seeds, used_features, scip_params, pool):
    tasks = [(seed, used_features, function_params, scip_params) for seed in seeds]
    results = pool.map(solve_one, tasks)
    for r in results:
        if not r["validity"]:
            print(f"  FAILED seed={r.get('seed')}: {r.get('error_info')}", flush=True)
            return FAILURE_PENALTY
    mean_nodes = numerical_mean_stable([r["nodes"] for r in results])
    print(f"  params={[round(p, 4) for p in function_params]} -> mean_nodes={mean_nodes:.2f}", flush=True)
    return mean_nodes


def tune(max_workers, n_calls, n_initial, random_state):
    _sf, used_features, _fp, bound_params = load_program(PROGRAM_PATH)
    scip_params = get_scip_params(time_limit=180)

    param_space = [Real(lo, hi) for lo, hi in bound_params]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=random_state)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        x0 = opt.ask(n_initial)
        y0 = [objective(x, TRAIN_SEEDS, used_features, scip_params, pool) for x in x0]
        opt.tell(x0, y0)
        for i in range(n_initial, n_calls):
            print(f"Original-on-WS tuning: iteration {i + 1}/{n_calls}", flush=True)
            next_x = opt.ask()
            f_val = objective(next_x, TRAIN_SEEDS, used_features, scip_params, pool)
            opt.tell(next_x, f_val)

    result = opt.get_result()
    return result.x, result.fun, used_features, bound_params


def write_program(theta, used_features, bound_params, out_path):
    with open(PROGRAM_PATH) as f:
        original_source = f.read()
    def_idx = original_source.index("def score_function")
    body = original_source[def_idx:]
    with open(out_path, "w") as f:
        f.write(f"USED_FEATURES = {used_features}\n")
        f.write(f"PARAMS = {[float(t) for t in theta]}\n")
        f.write(f"BOUNDS = {bound_params}\n")
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--n_calls", type=int, default=50)
    parser.add_argument("--n_initial", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--out_suffix", default="")
    args = parser.parse_args()

    print(f"Tuning Original structure on WS (n=200,k=20,p=0.25, natural D_sub={TRAIN_SEEDS}), "
          f"random_state={args.random_state}", flush=True)
    theta, fun, used_features, bound_params = tune(args.max_workers, args.n_calls, args.n_initial, args.random_state)
    print(f"\nDONE: best mean_nodes={fun:.2f}, theta={theta}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), f"program_original_ws_tuned{args.out_suffix}.py")
    write_program(theta, used_features, bound_params, out_path)
    print(f"Saved: {out_path}", flush=True)
