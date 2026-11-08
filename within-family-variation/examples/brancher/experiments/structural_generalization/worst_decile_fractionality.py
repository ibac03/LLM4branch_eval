"""
Follow-up to worst_decile_analysis.py: root_gap (vs the true optimum) was
the only property that split significantly, but it isn't computable before
solving. This checks a genuinely pre-solve proxy -- root LP fractionality
(mean distance from integral of the root relaxation's candidate variables)
-- available from literally the first observation, before any branching.
Same 100 seeds, same instances.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params

SG = os.path.dirname(__file__)
N_INSTANCES = 100
SEEDS = list(range(800000000, 800000000 + N_INSTANCES))
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
RESULTS_PATH = os.path.join(SG, "worst_decile_fractionality_results.json")


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def root_fractionality(seed):
    import ecole
    print(f"[seed={seed}] root-only probe", flush=True)
    instance = make_ba_instance(seed)
    # NodeBipartiteFlash (not plain NodeBipartite) to match this codebase's
    # feature indexing everywhere else: index 9 = "fractional part of LP
    # solution" under set_active_features([9]).
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    # Activating [9] alone returns garbage (uninitialized-looking values) --
    # Flash's lazy computation for index 9 ("fractional part") apparently
    # depends on index 8 ("LP solution value") being computed too; verified
    # with all-19-active that column 9 is only clean (974/974 finite,
    # correct [0,1] range) once 8 is also requested. Binary vars make 8 and
    # 9 numerically identical anyway (floor(x)=0 for x in [0,1)), so this
    # costs nothing.
    flash_node_obs.set_active_features([8, 9])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs},
        scip_params=get_scip_params(time_limit=60))
    env.seed(0)
    observation, action_set, _, done, info = env.reset(instance)
    frac_raw = observation["Node"].variable_features[action_set, 1]  # column 1 = feature 9
    # A true fractional part is mathematically bounded in [0,1) -- anything
    # outside that range is garbage regardless of whether it's literally
    # nan/inf (some variables return e.g. ~1e+305: finite, but nonsensical,
    # and large enough to overflow a plain sum even with nan_to_num applied).
    # Filter to the theoretically valid range instead of just isfinite().
    valid = np.isfinite(frac_raw) & (frac_raw >= -1e-6) & (frac_raw <= 1 + 1e-6)
    n_bad = int((~valid).sum())
    frac = frac_raw[valid]
    mean_frac = float(frac.mean()) if len(frac) else float("nan")
    frac_share = float((frac > 1e-6).mean()) if len(frac) else float("nan")
    print(f"[seed={seed}] done: mean_frac={mean_frac:.4f} frac_share={frac_share:.4f} "
          f"n_candidates={len(action_set)} n_bad(nan/inf)={n_bad}", flush=True)
    return {"seed": seed, "mean_root_fractionality": mean_frac,
            "root_fractional_share": frac_share, "n_root_candidates": len(action_set),
            "n_bad": n_bad}


if __name__ == "__main__":
    from multiprocessing import Pool
    print(f"=== root LP fractionality probe: {N_INSTANCES} BA instances (pre-solve proxy) ===", flush=True)
    with Pool(processes=6) as pool:
        records = pool.map(root_fractionality, SEEDS)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== FRACTIONALITY_PROBE_COMPLETE ===", flush=True)
