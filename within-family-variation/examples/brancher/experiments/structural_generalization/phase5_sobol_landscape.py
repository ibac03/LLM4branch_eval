"""
regular.md Phase 5: sample the parameter landscape via Sobol
quasi-random sampling (128 vectors in [0,2]^4, the IndSet policy's
bound space) and evaluate EVERY vector on BOTH BA and RR, to check for
a genuine Pareto tradeoff vs a shared-optimum landscape.

Domains (kept small/fast deliberately -- 128 vectors x 2 domains x 3
seeds = 768 solves):
  BA: affinity=5, n=400 (Category K's BA training tier -- already
      calibrated tractable, mean_nodes~1035)
  RR: degree=3, n=200 (Phase 1's calibrated RR regime, mean_nodes~4041)

Metric: mean_nodes over 3 held-out seeds (100,101,102 for BA;
2000,2001,2002 for RR -- disjoint from all earlier training/test seeds
used for these domains).
"""
import os
import sys
from functools import partial
from multiprocessing import Pool

import ecole
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import numerical_mean_stable, get_scip_params, load_program, normalize
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
N_SAMPLES = 32
BA_SEEDS = [100, 101, 102]
RR_SEEDS = [2000, 2001, 2002]
BA_N_NODES, BA_AFFINITY = 400, 5
RR_N_NODES, RR_DEGREE = 200, 3


def solve_one(args):
    domain, seed, used_features, function_params, scip_params = args
    score_function, _uf, _fp, _bp = load_program(PROGRAM_PATH)
    score_function = partial(score_function, params=function_params)

    if domain == "BA":
        instance = generate_ba_instance(seed=seed, n_nodes=BA_N_NODES, affinity=BA_AFFINITY)
    else:
        instance = generate_rr_instance(seed=seed, n_nodes=RR_N_NODES, degree=RR_DEGREE)

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
        return {"validity": True, "nodes": m.getNNodes(), "status": m.getStatus()}
    except Exception as e:
        return {"validity": False, "error_info": str(e), "status": "error"}


if __name__ == "__main__":
    _sf, used_features, _fp, bound_params = load_program(PROGRAM_PATH)
    scip_params = get_scip_params(time_limit=25)  # tight cap: adversarial Sobol corners can be pathological

    lo = np.array([b[0] for b in bound_params])
    hi = np.array([b[1] for b in bound_params])
    sampler = qmc.Sobol(d=len(bound_params), scramble=True, seed=42)
    unit_samples = sampler.random(N_SAMPLES)
    samples = qmc.scale(unit_samples, lo, hi)

    print(f"Phase 5: {N_SAMPLES} Sobol samples, BA(affinity={BA_AFFINITY},n={BA_N_NODES}) vs "
          f"RR(degree={RR_DEGREE},n={RR_N_NODES}), 3 seeds each, time_limit=60s", flush=True)

    with Pool(processes=4) as pool:
        for i, theta in enumerate(samples):
            theta = theta.tolist()
            ba_tasks = [("BA", s, used_features, theta, scip_params) for s in BA_SEEDS]
            rr_tasks = [("RR", s, used_features, theta, scip_params) for s in RR_SEEDS]
            ba_results = pool.map(solve_one, ba_tasks)
            rr_results = pool.map(solve_one, rr_tasks)

            ba_valid = all(r["validity"] for r in ba_results)
            rr_valid = all(r["validity"] for r in rr_results)
            ba_nodes = numerical_mean_stable([r["nodes"] for r in ba_results]) if ba_valid else float("nan")
            rr_nodes = numerical_mean_stable([r["nodes"] for r in rr_results]) if rr_valid else float("nan")
            ba_timeouts = sum(1 for r in ba_results if r.get("status") not in ("optimal",))
            rr_timeouts = sum(1 for r in rr_results if r.get("status") not in ("optimal",))

            print(f"SAMPLE {i}: theta={[round(t,4) for t in theta]} "
                  f"ba_timeouts={ba_timeouts}/3 rr_timeouts={rr_timeouts}/3 "
                  f"ba_mean_nodes={ba_nodes:.2f} rr_mean_nodes={rr_nodes:.2f}", flush=True)

    print("PHASE5_DONE", flush=True)
