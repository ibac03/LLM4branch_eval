"""
Quick calibration probe for sweep_ba_vs_rr_ws_difficulty.py's checkpoints.
n=500 (the originally planned checkpoint 1) turned out to be a massive
overshoot -- both rr-on-rr AND ba-on-rr timed out completely at the 900s
cap (0/2 optimal each), meaning it's universally hard, not BA-specific,
and useless as a comparison point. This probes smaller n_nodes steps
using each domain's OWN evolved policy (the cheapest, most relevant
reference) to find where solve time starts climbing but hasn't yet
blown past a sane cap -- a few seeds each, capped at 180s so a bad
candidate fails fast rather than eating another 900s smoke test.
"""
import sys
import os
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, geometric_mean_stable, normalize
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
RR_PROGRAM = os.path.join(SG, "rr_evolution_output", "best", "best_program.py")
WS_PROGRAM = os.path.join(SG, "ws_evolution_output", "best", "best_program.py")

CALIB_SEEDS = range(3)
CAP = 180  # fail fast -- we just want to see the climb, not solve hard instances


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
    rr_score_function, rr_used_features, rr_params, _ = load_program(RR_PROGRAM)
    rr_score_function = partial(rr_score_function, params=rr_params)

    ws_score_function, ws_used_features, ws_params, _ = load_program(WS_PROGRAM)
    ws_score_function = partial(ws_score_function, params=ws_params)

    print("=== RR calibration (own evolved program, degree=3 fixed) ===", flush=True)
    for n_nodes in [260, 280, 300, 330, 360, 400, 450]:
        all_opt, geo = probe(f"RR n={n_nodes},degree=3",
                              lambda seed, n=n_nodes: generate_rr_instance(seed=seed, n_nodes=n, degree=3),
                              rr_score_function, rr_used_features)
        if not all_opt:
            print(f"  -> RR n={n_nodes} already hitting the {CAP}s cap, stopping RR sweep here.", flush=True)
            break

    print("\n=== WS calibration (own evolved program, k=20,p=0.25 fixed) ===", flush=True)
    for n_nodes in [220, 250, 280, 310, 350, 400, 450]:
        all_opt, geo = probe(f"WS n={n_nodes},k=20,p=0.25",
                              lambda seed, n=n_nodes: generate_ws_instance(seed=seed, n_nodes=n, k=20, p=0.25),
                              ws_score_function, ws_used_features)
        if not all_opt:
            print(f"  -> WS n={n_nodes} already hitting the {CAP}s cap, stopping WS sweep here.", flush=True)
            break
