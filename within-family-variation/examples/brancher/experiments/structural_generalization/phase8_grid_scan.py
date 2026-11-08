"""
Grid scan over the 2 LIVE parameters (fractionality, constraint_count --
params[2] and params[3]; params[0]/params[1] are confirmed dead for
unweighted MIS, fixed at 0) on BA and RR, to directly map
L_BA(theta_frac, theta_count) and L_RR(theta_frac, theta_count) and
check for a genuine Pareto/optimum conflict vs a shared optimum.

11x11 grid over [0,2]^2 = 121 points, 8 seeds per domain per point.
Domains: BA (affinity=5, n=400 -- calibrated tractable), RR (n=250,
degree=3 -- calibrated ~29s geomean, the new default).
Metric: mean_nodes (numerical/arithmetic mean, matching the BO tuning
scripts' convention).
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

GRID_N = 11  # 11x11 = 121 points over [0,2]^2
N_SEEDS = 8
BA_N_NODES, BA_AFFINITY = 400, 5
RR_N_NODES, RR_DEGREE = 250, 3
BA_SEEDS = list(range(200, 200 + N_SEEDS))   # fresh, disjoint from all earlier BA seed ranges used
RR_SEEDS = list(range(3000, 3000 + N_SEEDS))  # fresh, disjoint from all earlier RR seed ranges used
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
    grid_vals = np.linspace(0, 2, GRID_N)
    print(f"Phase 8: {GRID_N}x{GRID_N} grid over [0,2]^2 (theta_frac, theta_count), "
          f"{N_SEEDS} seeds/domain, BA(affinity={BA_AFFINITY},n={BA_N_NODES}) vs "
          f"RR(degree={RR_DEGREE},n={RR_N_NODES}), time_cap={TIME_CAP}s", flush=True)

    with Pool(processes=4) as pool:
        for i, theta_frac in enumerate(grid_vals):
            for j, theta_count in enumerate(grid_vals):
                ba_tasks = [("BA", s, theta_frac, theta_count) for s in BA_SEEDS]
                rr_tasks = [("RR", s, theta_frac, theta_count) for s in RR_SEEDS]
                ba_results = pool.map(solve_one, ba_tasks)
                rr_results = pool.map(solve_one, rr_tasks)

                ba_timeouts = sum(1 for r in ba_results if r.get("status") not in ("optimal",))
                rr_timeouts = sum(1 for r in rr_results if r.get("status") not in ("optimal",))
                ba_nodes = numerical_mean_stable([r["nodes"] for r in ba_results if r["validity"]])
                rr_nodes = numerical_mean_stable([r["nodes"] for r in rr_results if r["validity"]])

                print(f"GRID i={i} j={j} theta_frac={theta_frac:.3f} theta_count={theta_count:.3f} "
                      f"ba_timeouts={ba_timeouts}/{N_SEEDS} rr_timeouts={rr_timeouts}/{N_SEEDS} "
                      f"ba_mean_nodes={ba_nodes:.2f} rr_mean_nodes={rr_nodes:.2f}", flush=True)

    print("PHASE8_GRID_DONE", flush=True)
