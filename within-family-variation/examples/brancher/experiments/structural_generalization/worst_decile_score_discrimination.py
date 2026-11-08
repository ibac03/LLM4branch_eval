"""
Follow-up #3: a property of the POLICY-INSTANCE interaction, not the graph
alone. At the root, the BA policy scores every candidate variable and
branches on the highest. If those scores come out nearly identical across
many candidates, the policy can't really tell them apart on this instance --
a plausible, different mechanism for "does worse here" than any graph
property. Root-only probe (same preprocessing pipeline -- nan_to_num +
normalize -- this codebase always uses before scoring), no branching needed.
Same 100 seeds as every other check in this analysis.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, normalize

SG = os.path.dirname(__file__)
BA_PROGRAM = os.path.join(SG, "..", "..", "program", "indset", "program.py")
N_INSTANCES = 100
SEEDS = list(range(800000000, 800000000 + N_INSTANCES))
BA_PARAMS = {"n_nodes": 1000, "affinity": 4}
RESULTS_PATH = os.path.join(SG, "worst_decile_score_discrimination_results.json")


def make_ba_instance(seed):
    import ecole
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_PARAMS["n_nodes"], affinity=BA_PARAMS["affinity"],
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def score_discrimination(seed):
    import ecole
    from functools import partial
    print(f"[seed={seed}] root-only score probe", flush=True)

    score_function, used_features, function_params, _ = load_program(BA_PROGRAM)
    score_function = partial(score_function, params=function_params)

    instance = make_ba_instance(seed)
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=60))
    env.seed(0)
    observation, action_set, _, done, info = env.reset(instance)

    variable_features = np.concatenate(
        (observation["Node"].variable_features[action_set, :],
         observation["Khalil"].features[action_set, :]), axis=1)
    variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
    variable_features = normalize(variable_features, used_features)
    scores = score_function(variable_features)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    n_bad = int((~valid).sum())
    scores = scores[valid]

    sorted_scores = np.sort(scores)[::-1]
    top1 = float(sorted_scores[0])
    top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else top1
    score_std = float(scores.std())
    score_range = float(scores.max() - scores.min())
    gap = top1 - top2
    gap_normalized = float(gap / score_std) if score_std > 0 else float("nan")

    rec = {
        "seed": seed, "n_candidates": len(scores), "n_bad": n_bad,
        "score_std": score_std, "score_range": score_range,
        "top1_top2_gap": float(gap), "top1_top2_gap_normalized": gap_normalized,
    }
    print(f"[seed={seed}] done: std={score_std:.4f} range={score_range:.4f} "
          f"gap={gap:.4f} gap_norm={gap_normalized:.4f} n={len(scores)} n_bad={n_bad}", flush=True)
    return rec


if __name__ == "__main__":
    from multiprocessing import Pool
    print(f"=== root score-discrimination probe: {N_INSTANCES} BA instances ===", flush=True)
    with Pool(processes=6) as pool:
        records = pool.map(score_discrimination, SEEDS)
    records.sort(key=lambda r: r["seed"])
    json.dump(records, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    print("\n=== SCORE_DISCRIMINATION_COMPLETE ===", flush=True)
