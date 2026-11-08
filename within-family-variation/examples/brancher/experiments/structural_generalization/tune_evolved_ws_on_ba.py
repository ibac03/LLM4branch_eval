"""
Take the LLM-evolved WS program's exact feature set/structure
(USED_FEATURES=[22,37,38,43,52,54,55,85], the pseudocost/fractional/
clustering formula) and re-optimize its 6 params specifically for BA via
the same BO recipe used throughout this project.

Triggered by screen_evolved_ws.py: native Evolved-WS params hit the
120s cap on 4/4 BA screen seeds (0/4 optimal), a much worse pattern than
RPB's own 2/4 timeouts on the same seeds -- a clear "failing badly"
signal per the established retune-trigger rule, unlike RR where native
params passed (0.95x speedup vs RPB, all-optimal).
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
from category_k_indset_affinity import generate_instance as generate_ba_instance

EVOLVED_PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "ws_evolution_output", "best", "best_program.py")
TRAIN_SEEDS = [0, 1, 2, 3]  # same natural D_sub=4 convention as every other tuning run this session
BA_N_NODES, BA_AFFINITY = 400, 5  # calibrated tractable BA training tier, used throughout this session
FAILURE_PENALTY = 1e9


def solve_one(args):
    seed, used_features, function_params, scip_params = args
    from functools import partial as _partial
    score_function, _uf, _fp, _bp = load_program(EVOLVED_PROGRAM_PATH)
    score_function = _partial(score_function, params=function_params)

    instance = generate_ba_instance(seed=seed, n_nodes=BA_N_NODES, affinity=BA_AFFINITY)

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
    _sf, used_features, _fp, bound_params = load_program(EVOLVED_PROGRAM_PATH)
    scip_params = get_scip_params(time_limit=180)

    param_space = [Real(lo, hi) for lo, hi in bound_params]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=random_state)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        x0 = opt.ask(n_initial)
        y0 = [objective(x, TRAIN_SEEDS, used_features, scip_params, pool) for x in x0]
        opt.tell(x0, y0)
        for i in range(n_initial, n_calls):
            print(f"Evolved-WS-on-BA tuning: iteration {i + 1}/{n_calls}", flush=True)
            next_x = opt.ask()
            f_val = objective(next_x, TRAIN_SEEDS, used_features, scip_params, pool)
            opt.tell(next_x, f_val)

    result = opt.get_result()
    return result.x, result.fun, used_features, bound_params


def write_program(theta, used_features, bound_params, out_path):
    with open(EVOLVED_PROGRAM_PATH) as f:
        original_source = f.read()
    # Find the actual score_function definition rather than assuming a fixed
    # 3-line header (evolved programs may have extra lines before
    # USED_FEATURES, e.g. "import numpy as np", unlike program/*/program.py's
    # bare 3-line header -- a fixed split() here left stale duplicate
    # PARAMS/BOUNDS lines baked into earlier runs' output, see git history).
    def_idx = original_source.index("def score_function")
    body = original_source[def_idx:]
    # tune_evolved_rr_on_ba.py's version of this dropped the "import numpy
    # as np" line entirely (harmless there since that structure's body used
    # only +/* operators), but Evolved-WS's body calls np.sqrt/np.log1p
    # directly -- dropping the import silently broke every solve with a
    # NameError. tester_parallel.py's solver() swallows exceptions without
    # setting validity=False, so this failed silently as "Proc Time: 0.0"
    # instead of erroring -- always re-add the import explicitly here.
    with open(out_path, "w") as f:
        f.write("import numpy as np\n")
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

    print(f"Tuning Evolved-WS structure on BA (affinity={BA_AFFINITY}, n={BA_N_NODES}, "
          f"natural D_sub={TRAIN_SEEDS}), random_state={args.random_state}", flush=True)
    theta, fun, used_features, bound_params = tune(args.max_workers, args.n_calls, args.n_initial, args.random_state)
    print(f"\nDONE: best mean_nodes={fun:.2f}, theta={theta}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), f"program_evolved_ws_ba_tuned{args.out_suffix}.py")
    write_program(theta, used_features, bound_params, out_path)
    print(f"Saved: {out_path}", flush=True)
