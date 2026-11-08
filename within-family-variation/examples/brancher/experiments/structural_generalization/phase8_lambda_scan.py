"""
1D mixture scan, replacing the 2D grid: since branching only ever uses
argmax(score) and both live parameters (fractionality, constraint_count)
are non-negative weights on the same linear score, ranking is invariant
to positive rescaling -- (theta_frac, theta_count) and
(k*theta_frac, k*theta_count) are behaviorally identical for any k>0.
So the true search space collapses to a single ratio, parameterized as:

    theta_frac = lambda,  theta_count = 1 - lambda,  lambda in [0,1]

lambda=0   -> pure constraint_count (degree-only)
lambda=1   -> pure fractionality
lambda=0.5 -> equal mix

21 lambda values, 8 seeds/domain, BA(affinity=5,n=400) vs
RR(degree=3,n=250) = 21*8*2 = 336 solves total (vs 1936 for the 2D grid).
"""
import os
import sys
from functools import partial
from multiprocessing import Pool

import ecole
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import numerical_mean_stable, get_scip_params, load_program, normalize
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
_sf, used_features, _fp, _bp = load_program(PROGRAM_PATH)

N_LAMBDA = 21
N_SEEDS = 8
BA_N_NODES, BA_AFFINITY = 400, 5
RR_N_NODES, RR_DEGREE = 250, 3
BA_SEEDS = list(range(200, 200 + N_SEEDS))
RR_SEEDS = list(range(3000, 3000 + N_SEEDS))
TIME_CAP = 60


def solve_one(args):
    domain, seed, theta_frac, theta_count = args
    function_params = [0.0, 0.0, theta_frac, theta_count]
    score_function, _uf, _fp2, _bp2 = load_program(PROGRAM_PATH)
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
        scip_params=get_scip_params(time_limit=TIME_CAP),
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
    lambdas = np.linspace(0, 1, N_LAMBDA)
    print(f"Phase 8 (1D): {N_LAMBDA} lambda values, {N_SEEDS} seeds/domain, "
          f"BA(affinity={BA_AFFINITY},n={BA_N_NODES}) vs RR(degree={RR_DEGREE},n={RR_N_NODES}), "
          f"time_cap={TIME_CAP}s", flush=True)

    with Pool(processes=4) as pool:
        for i, lam in enumerate(lambdas):
            theta_frac, theta_count = float(lam), float(1 - lam)
            ba_tasks = [("BA", s, theta_frac, theta_count) for s in BA_SEEDS]
            rr_tasks = [("RR", s, theta_frac, theta_count) for s in RR_SEEDS]
            ba_results = pool.map(solve_one, ba_tasks)
            rr_results = pool.map(solve_one, rr_tasks)

            ba_timeouts = sum(1 for r in ba_results if r.get("status") not in ("optimal",))
            rr_timeouts = sum(1 for r in rr_results if r.get("status") not in ("optimal",))
            ba_nodes = numerical_mean_stable([r["nodes"] for r in ba_results if r["validity"]])
            rr_nodes = numerical_mean_stable([r["nodes"] for r in rr_results if r["validity"]])

            print(f"LAMBDA i={i} lambda={lam:.3f} theta_frac={theta_frac:.3f} theta_count={theta_count:.3f} "
                  f"ba_timeouts={ba_timeouts}/{N_SEEDS} rr_timeouts={rr_timeouts}/{N_SEEDS} "
                  f"ba_mean_nodes={ba_nodes:.2f} rr_mean_nodes={rr_nodes:.2f}", flush=True)

    print("PHASE8_LAMBDA_DONE", flush=True)
