"""
regular.md Phase 6: mechanistic feature-contribution check. At several
branching nodes on BA vs RR instances (using Original's released
params), log the per-candidate contribution of each of the 4 score
terms and their std across candidate variables at that node --
especially constraint_count's term, to see directly whether degree
carries real information at the moment of branching (not just in the
initial, unbranched graph -- SCIP presolve/search can change things, so
this measures it directly rather than assuming).
"""
import sys
import os
from functools import partial

import numpy as np
import ecole

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, load_program, normalize
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
_sf, used_features, function_params, _bp = load_program(PROGRAM_PATH)
N_NODES_TO_LOG = 5  # log stats at the first 5 branching decisions


def log_run(domain, instance, seed):
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
    while not done and step < N_NODES_TO_LOG:
        variable_features = np.concatenate(
            (observation["Node"].variable_features[action_set, :],
             observation["Khalil"].features[action_set, :]), axis=1)
        variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
        variable_features = normalize(variable_features, used_features)

        obj_coeff = variable_features[:, 0]
        reduced_cost = variable_features[:, 7]
        fractional_part = variable_features[:, 9]
        constraint_count = variable_features[:, 22]
        branching_preference = 4.0 * fractional_part * (1.0 - fractional_part)

        obj_term = function_params[0] * obj_coeff
        rc_term = function_params[1] * reduced_cost
        frac_term = function_params[2] * branching_preference
        count_term = function_params[3] * constraint_count

        print(f"{domain} seed={seed} node={step}: n_candidates={len(action_set)} "
              f"std(obj_term)={obj_term.std():.4f} std(rc_term)={rc_term.std():.4f} "
              f"std(frac_term)={frac_term.std():.4f} std(count_term)={count_term.std():.4f} "
              f"raw_constraint_count_std={constraint_count.std():.4f}", flush=True)

        scores = (obj_term + rc_term + frac_term + count_term)
        action = action_set[scores.argmax()]
        observation, action_set, _, done, info = env.step(action)
        step += 1


if __name__ == "__main__":
    print("=== Phase 6: feature contribution std, BA vs RR (Original's params) ===", flush=True)
    for seed in range(3):
        instance = generate_ba_instance(seed=seed, n_nodes=1000, affinity=4)
        log_run("BA", instance, seed)
    for seed in range(3):
        instance = generate_rr_instance(seed=seed, n_nodes=200, degree=3)
        log_run("RR", instance, seed)
    print("PHASE6_DONE", flush=True)
