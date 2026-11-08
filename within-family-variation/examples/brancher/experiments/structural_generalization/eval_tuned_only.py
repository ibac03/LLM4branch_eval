"""
Evaluate ONLY the Category-G-tuned policy on the natural medium test set
(seeds 300-379) -- RPB and original LLM4Branch are already being evaluated
separately on the same seeds, so this just adds the third arm.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import ecole

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, load_program, normalize, geometric_mean_stable
from category_g_multicover_generator import generate_instance
from functools import partial

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "program_category_g_tuned.py")
TIME_LIMIT = 120


def solve_llm4branch(instance, score_function, used_features):
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
    parser.add_argument("--num_instances", type=int, default=80)
    parser.add_argument("--seed_start", type=int, default=300)
    parser.add_argument("--seed_list_file", default=None,
                         help="path to a file with one seed per line -- overrides seed_start/num_instances")
    parser.add_argument("--out_suffix", default="")
    parser.add_argument("--program_path", default=None, help="override PROGRAM_PATH")
    args = parser.parse_args()

    if args.program_path:
        PROGRAM_PATH = args.program_path

    if args.seed_list_file:
        with open(args.seed_list_file) as f:
            seed_range = [int(line.strip()) for line in f if line.strip()]
    else:
        seed_range = list(range(args.seed_start, args.seed_start + args.num_instances))

    score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
    score_function = partial(score_function, params=function_params)

    results_path = os.path.join(os.path.dirname(__file__), f"tuned_results{args.out_suffix}.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "tuned_time", "tuned_nodes", "tuned_status"])

        times, nodes_l, timeouts = [], [], 0
        for seed in range(args.seed_start, args.seed_start + args.num_instances):
            model, _ = generate_instance(seed=seed)
            instance = ecole.scip.Model.from_pyscipopt(model)
            t, n, status = solve_llm4branch(instance, score_function, used_features)
            times.append(t); nodes_l.append(n)
            if status != "optimal":
                timeouts += 1
            print(f"seed={seed}: Tuned(time={t:.2f}, nodes={n}, status={status})", flush=True)
            writer.writerow([seed, t, n, status])
            f.flush()

    print(f"\nTuned: geomean_time={geometric_mean_stable(times):.2f}, geomean_nodes={geometric_mean_stable(nodes_l):.1f}, timeouts={timeouts}")
    print(f"Saved: {results_path}")
