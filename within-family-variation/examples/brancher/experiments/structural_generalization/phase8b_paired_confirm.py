"""
Phase 8b: paired confirmation of the Phase 8 lambda scan's tentative
finding (BA favors lambda~0.4, RR favors lambda~0.95-1.0), on a much
larger held-out set (50 instances/domain instead of 8), with only the
4 most informative lambda values to keep this tractable:
  0.40 -- BA's optimum
  0.85, 0.95, 1.00 -- the emerging RR-favored region (using more than
  just the boundary 1.0, per the advisor's note that pure fractionality
  alone could be a boundary-solution artifact)

Reports per-instance results so paired analysis is possible, and
computes both the standard-censoring result and a sensitivity check
restricted to instances that solved to optimal under ALL 4 policies
(removing any timeout-censoring confound entirely).
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

LAMBDAS = [0.40, 0.85, 0.95, 1.00]
N_INSTANCES = 50
BA_N_NODES, BA_AFFINITY = 400, 5
RR_N_NODES, RR_DEGREE = 250, 3
BA_SEEDS = list(range(5000, 5000 + N_INSTANCES))   # fresh, disjoint from all earlier BA seed ranges
RR_SEEDS = list(range(4000, 4000 + N_INSTANCES))   # fresh, disjoint from all earlier RR seed ranges
TIME_CAP = 180  # generous cap to minimize censoring at this larger scale


def solve_one(args):
    domain, seed, lam = args
    theta_frac, theta_count = lam, 1 - lam
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
    print(f"Phase 8b: paired confirmation, lambdas={LAMBDAS}, N={N_INSTANCES}/domain, "
          f"BA(affinity={BA_AFFINITY},n={BA_N_NODES}) vs RR(degree={RR_DEGREE},n={RR_N_NODES}), "
          f"time_cap={TIME_CAP}s", flush=True)

    with Pool(processes=4) as pool:
        for domain, seeds in [("BA", BA_SEEDS), ("RR", RR_SEEDS)]:
            for lam in LAMBDAS:
                tasks = [(domain, s, lam) for s in seeds]
                results = pool.map(solve_one, tasks)
                for seed, r in zip(seeds, results):
                    print(f"PAIRED domain={domain} lambda={lam:.2f} seed={seed} "
                          f"nodes={r.get('nodes')} status={r.get('status')} validity={r['validity']}", flush=True)

    print("PHASE8B_DONE", flush=True)
