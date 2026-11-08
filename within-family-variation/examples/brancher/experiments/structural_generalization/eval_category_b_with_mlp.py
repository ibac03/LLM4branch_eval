"""
Evaluate LLM4Branch's released policy AND the MLP baseline together on the
same N Category-B instances (clustered/modular structure), at the calibrated
medium setting (500x500, n_clusters=4, local_density=0.08). RPB is skipped
here -- its results on these same seeds are already in category_b_results.csv.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import ecole
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, load_program, normalize, geometric_mean_stable
from baselines.MLP.model import MLPPolicy
from category_b_generator import generate_instance
from functools import partial

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "setcover", "program.py")
MLP_CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "MLP", "checkpoints", "epoch_010.pt")
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


def solve_mlp(instance, model):
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_node_obs.set_active_features(list(range(19)))
    flash_khalil_obs.set_active_features(list(range(72)))
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=TIME_LIMIT),
    )
    env.seed(0)
    t0 = time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    with torch.no_grad():
        while not done:
            variable_features = np.concatenate(
                (observation["Node"].variable_features[action_set, :],
                 observation["Khalil"].features[action_set, :]), axis=1)
            variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
            variable_features = normalize(variable_features, list(range(91)))
            scores = model(torch.from_numpy(variable_features.astype(np.float32))).squeeze(-1).numpy()
            action = action_set[scores.argmax()]
            observation, action_set, _, done, info = env.step(action)
    proc_time = time.process_time() - t0
    m = env.model.as_pyscipopt()
    return proc_time, m.getNNodes(), m.getStatus()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int, default=25)
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
    score_function = partial(score_function, params=function_params)

    mlp_model = MLPPolicy()
    ckpt = torch.load(MLP_CKPT_PATH, map_location="cpu", weights_only=False)
    mlp_model.load_state_dict(ckpt["model_state_dict"])
    mlp_model.eval()

    results_path = os.path.join(os.path.dirname(__file__), "category_b_with_mlp_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "llm4branch_time", "llm4branch_nodes", "llm4branch_status",
                          "mlp_time", "mlp_nodes", "mlp_status"])

        llm_times, llm_nodes_l = [], []
        mlp_times, mlp_nodes_l = [], []
        llm_timeouts, mlp_timeouts = 0, 0
        wins_llm, wins_mlp = 0, 0

        for seed in range(args.seed_start, args.seed_start + args.num_instances):
            model2, _ = generate_instance(seed=seed)
            instance2 = ecole.scip.Model.from_pyscipopt(model2)
            llm_time, llm_nodes, llm_status = solve_llm4branch(instance2, score_function, used_features)

            model3, _ = generate_instance(seed=seed)
            instance3 = ecole.scip.Model.from_pyscipopt(model3)
            mlp_time, mlp_nodes, mlp_status = solve_mlp(instance3, mlp_model)

            llm_times.append(llm_time); llm_nodes_l.append(llm_nodes)
            mlp_times.append(mlp_time); mlp_nodes_l.append(mlp_nodes)
            if llm_status != "optimal":
                llm_timeouts += 1
            if mlp_status != "optimal":
                mlp_timeouts += 1
            if llm_time < mlp_time:
                wins_llm += 1
            else:
                wins_mlp += 1

            print(f"seed={seed}: LLM4Branch(time={llm_time:.2f}, nodes={llm_nodes}, status={llm_status}) "
                  f"MLP(time={mlp_time:.2f}, nodes={mlp_nodes}, status={mlp_status})", flush=True)
            writer.writerow([seed, llm_time, llm_nodes, llm_status, mlp_time, mlp_nodes, mlp_status])
            f.flush()

    print(f"\nLLM4Branch: geomean_time={geometric_mean_stable(llm_times):.2f}, geomean_nodes={geometric_mean_stable(llm_nodes_l):.1f}, timeouts={llm_timeouts}")
    print(f"MLP: geomean_time={geometric_mean_stable(mlp_times):.2f}, geomean_nodes={geometric_mean_stable(mlp_nodes_l):.1f}, timeouts={mlp_timeouts}")
    print(f"Wins -- LLM4Branch: {wins_llm}, MLP: {wins_mlp}")
    print(f"Saved: {results_path}")
