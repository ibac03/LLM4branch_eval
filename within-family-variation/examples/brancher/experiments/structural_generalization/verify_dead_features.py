"""
Rigorous verification of Phase 6's claim: obj_coeff and reduced_cost are
structurally constant across candidates for unweighted MIS, across BA,
ER, and RR, many more nodes deep, many more seeds -- not just an
artifact of a shallow/few-seed check.
"""
import sys
import os

import numpy as np
import ecole

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, load_program, normalize
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance
from category_k_er_indset_large import generate_instance as generate_er_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
_sf, used_features, function_params, _bp = load_program(PROGRAM_PATH)
N_NODES_TO_CHECK = 30  # much deeper than Phase 6's original 5


def check_run(domain, instance, seed):
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=60),
    )
    env.seed(0)
    observation, action_set, _, done, info = env.reset(instance)
    step = 0
    max_obj_std, max_rc_std = 0.0, 0.0
    while not done and step < N_NODES_TO_CHECK:
        variable_features = np.concatenate(
            (observation["Node"].variable_features[action_set, :],
             observation["Khalil"].features[action_set, :]), axis=1)
        variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
        # Check RAW (pre-normalization) values too, to rule out a normalize()-induced artifact
        raw_obj = variable_features[:, 0].copy()
        raw_rc = variable_features[:, 7].copy()
        variable_features = normalize(variable_features, used_features)

        obj_coeff = variable_features[:, 0]
        reduced_cost = variable_features[:, 7]
        fractional_part = variable_features[:, 9]
        constraint_count = variable_features[:, 22]
        branching_preference = 4.0 * fractional_part * (1.0 - fractional_part)

        max_obj_std = max(max_obj_std, obj_coeff.std())
        max_rc_std = max(max_rc_std, reduced_cost.std())

        if step in (0, N_NODES_TO_CHECK - 1):
            print(f"{domain} seed={seed} node={step}: raw_obj_std={raw_obj.std():.6f} "
                  f"raw_rc_std={raw_rc.std():.6f} raw_rc_sample={raw_rc[:3]} "
                  f"norm_obj_std={obj_coeff.std():.6f} norm_rc_std={reduced_cost.std():.6f}", flush=True)

        score = (function_params[0] * obj_coeff + function_params[1] * reduced_cost +
                 function_params[2] * branching_preference + function_params[3] * constraint_count)
        action = action_set[score.argmax()]
        observation, action_set, _, done, info = env.step(action)
        step += 1

    print(f"{domain} seed={seed}: max_obj_std_over_{step}_nodes={max_obj_std:.6f} "
          f"max_rc_std_over_{step}_nodes={max_rc_std:.6f}", flush=True)


if __name__ == "__main__":
    print("=== Verifying dead-feature claim: deeper (30 nodes), more seeds, BA+ER+RR ===", flush=True)
    for seed in range(5):
        instance = generate_ba_instance(seed=seed, n_nodes=1000, affinity=4)
        check_run("BA", instance, seed)
    for seed in range(5):
        instance = generate_rr_instance(seed=seed, n_nodes=250, degree=3)
        check_run("RR", instance, seed)
    for seed in range(5):
        instance = generate_er_instance(seed=seed)
        check_run("ER", instance, seed)
    print("VERIFY_DEAD_FEATURES_DONE", flush=True)
