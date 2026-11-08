"""
Evaluate ONE arm on a fresh, non-overlapping batch of Category G medium
instances (seeds 1000-1149, 150 instances -- prior batches used 0-379).
"""
import argparse
import csv
import os
import sys
import time

import ecole
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, load_program, normalize, geometric_mean_stable
from category_g_multicover_generator import generate_instance
from functools import partial

FRESH_SEED_START = 1000
FRESH_NUM = 150
TIME_LIMIT = 120


def solve_rpb(instance):
    scip_params = {**get_scip_params(time_limit=TIME_LIMIT), "branching/relpscost/priority": 9999999}
    env = ecole.environment.Configuring(scip_params=scip_params)
    env.seed(0)
    t0 = time.process_time()
    env.reset(instance)
    env.step({})
    proc_time = time.process_time() - t0
    m = env.model.as_pyscipopt()
    return proc_time, m.getNNodes(), m.getStatus()


def solve_policy(instance, score_function, used_features):
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=TIME_LIMIT),
    )
    env.seed(0)
    t0 = time.process_time()
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
    proc_time = time.process_time() - t0
    m = env.model.as_pyscipopt()
    return proc_time, m.getNNodes(), m.getStatus()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm_name", required=True)
    parser.add_argument("--program_path", default=None, help="omit for RPB")
    parser.add_argument("--seed_start", type=int, default=FRESH_SEED_START)
    parser.add_argument("--num_instances", type=int, default=FRESH_NUM)
    args = parser.parse_args()

    is_rpb = args.program_path is None
    if not is_rpb:
        score_function, used_features, function_params, _ = load_program(args.program_path)
        score_function = partial(score_function, params=function_params)

    out_dir = os.path.join(os.path.dirname(__file__), "categoryg_fresh")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"{args.arm_name}_fresh.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "proc_time", "nodes", "status"])
        times, nodes_l, timeouts = [], [], 0
        for seed in range(args.seed_start, args.seed_start + args.num_instances):
            model, _ = generate_instance(seed=seed)
            instance = ecole.scip.Model.from_pyscipopt(model)
            if is_rpb:
                t, n, status = solve_rpb(instance)
            else:
                t, n, status = solve_policy(instance, score_function, used_features)
            times.append(t); nodes_l.append(n)
            if status != "optimal":
                timeouts += 1
            print(f"seed={seed}: {args.arm_name}(time={t:.2f}, nodes={n}, status={status})", flush=True)
            writer.writerow([seed, t, n, status])
            f.flush()

    print(f"\n{args.arm_name}: geomean_time={geometric_mean_stable(times):.2f}, "
          f"geomean_nodes={geometric_mean_stable(nodes_l):.1f}, timeouts={timeouts}")
    print(f"Saved: {results_path}")
