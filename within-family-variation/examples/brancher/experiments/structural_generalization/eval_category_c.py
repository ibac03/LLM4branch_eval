"""
Evaluate RPB and LLM4Branch's released policy on N Category-C instances
(heavy-tailed column degree + bottleneck rows), to see whether LLM4Branch
struggles relative to RPB on this structurally different instance family --
not just whether it's harder for everyone.
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
from category_c_generator import generate_instance
from functools import partial

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "setcover", "program.py")
TIME_LIMIT = 120  # seconds -- Category B's clustered/modular structure can trigger
                  # pathological branch-and-bound symmetry (near-identical independent
                  # cluster subproblems); cap hard so a batch can't stall on one seed.


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
    parser.add_argument("--num_instances", type=int, default=30)
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
    score_function = partial(score_function, params=function_params)

    results_path = os.path.join(os.path.dirname(__file__), "category_c_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "rpb_time", "rpb_nodes", "rpb_status",
                          "llm4branch_time", "llm4branch_nodes", "llm4branch_status"])

        rpb_times, rpb_nodes_l, llm_times, llm_nodes_l = [], [], [], []
        wins_rpb, wins_llm = 0, 0
        rpb_timeouts, llm_timeouts = 0, 0

        for seed in range(args.seed_start, args.seed_start + args.num_instances):
            model, _ = generate_instance(seed=seed)
            instance = ecole.scip.Model.from_pyscipopt(model)
            rpb_time, rpb_nodes, rpb_status = solve_rpb(instance)

            model2, _ = generate_instance(seed=seed)  # fresh model (SCIP models are consumed by solving)
            instance2 = ecole.scip.Model.from_pyscipopt(model2)
            llm_time, llm_nodes, llm_status = solve_llm4branch(instance2, score_function, used_features)

            rpb_times.append(rpb_time); rpb_nodes_l.append(rpb_nodes)
            llm_times.append(llm_time); llm_nodes_l.append(llm_nodes)
            if rpb_status != "optimal":
                rpb_timeouts += 1
            if llm_status != "optimal":
                llm_timeouts += 1
            if rpb_time < llm_time:
                wins_rpb += 1
            else:
                wins_llm += 1

            print(f"seed={seed}: RPB(time={rpb_time:.2f}, nodes={rpb_nodes}, status={rpb_status}) "
                  f"LLM4Branch(time={llm_time:.2f}, nodes={llm_nodes}, status={llm_status})", flush=True)
            writer.writerow([seed, rpb_time, rpb_nodes, rpb_status, llm_time, llm_nodes, llm_status])
            f.flush()

    print(f"\nRPB: geomean_time={geometric_mean_stable(rpb_times):.2f}, geomean_nodes={geometric_mean_stable(rpb_nodes_l):.1f}, timeouts={rpb_timeouts}")
    print(f"LLM4Branch: geomean_time={geometric_mean_stable(llm_times):.2f}, geomean_nodes={geometric_mean_stable(llm_nodes_l):.1f}, timeouts={llm_timeouts}")
    print(f"Wins -- RPB: {wins_rpb}, LLM4Branch: {wins_llm}")
    print(f"Saved: {results_path}")
