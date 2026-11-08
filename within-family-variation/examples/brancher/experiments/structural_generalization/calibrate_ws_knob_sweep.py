"""
Hypothesis 1 test: does WS's ba-vs-own gap grow if difficulty is raised
via a DIFFERENT knob (k, the ring-lattice neighborhood size -- WS's
analog to RR's "degree") at FIXED n_nodes=200, instead of raising
n_nodes the way the first sweep did (where the gap stayed flat at
2.16x/2.13x)? Calibrates k upward (p=0.25 fixed) using WS's own evolved
policy as the reference, same pattern as calibrate_sweep_checkpoints.py.
"""
import sys
import os
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, geometric_mean_stable, normalize
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
WS_PROGRAM = os.path.join(SG, "ws_evolution_output", "best", "best_program.py")

CALIB_SEEDS = range(3)
CAP = 180


def solve_one(instance, score_function, used_features, time_limit):
    import ecole
    import time as _time
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=time_limit),
    )
    env.seed(0)
    t0 = _time.process_time()
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
    proc_time = _time.process_time() - t0
    m = env.model.as_pyscipopt()
    return proc_time, m.getNNodes(), m.getStatus()


def probe(label, make_instance, score_function, used_features):
    times, nodes_list, statuses = [], [], []
    for seed in CALIB_SEEDS:
        instance = make_instance(seed)
        t, n, s = solve_one(instance, score_function, used_features, CAP)
        times.append(t)
        nodes_list.append(n)
        statuses.append(s)
    all_optimal = all(s == "optimal" for s in statuses)
    print(f"{label}: times={[round(t, 2) for t in times]} statuses={statuses} "
          f"all_optimal={all_optimal} geomean={geometric_mean_stable(times):.2f} "
          f"mean_nodes={np.mean(nodes_list):.1f}", flush=True)
    return all_optimal, geometric_mean_stable(times)


if __name__ == "__main__":
    ws_score_function, ws_used_features, ws_params, _ = load_program(WS_PROGRAM)
    ws_score_function = partial(ws_score_function, params=ws_params)

    print("=== WS k-knob calibration (own evolved program, n=200,p=0.25 fixed) ===", flush=True)
    for k in [25, 30, 35, 40, 50, 60, 70]:
        all_opt, geo = probe(f"WS n=200,k={k},p=0.25",
                              lambda seed, kk=k: generate_ws_instance(seed=seed, n_nodes=200, k=kk, p=0.25),
                              ws_score_function, ws_used_features)
        if not all_opt:
            print(f"  -> WS k={k} already hitting the {CAP}s cap, stopping sweep here.", flush=True)
            break
