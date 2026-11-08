"""
Sanity check: does tune_category_k.py's solve_one() (used inside the BO
loop) produce the SAME result as eval_category_b.py's solve_llm4branch()
(used in calibration) for the SAME instance and the SAME (Original)
params? If they match, the tuning harness's instance generation /
feature slicing / normalization is wired correctly and the earlier weak
result is a genuine finding, not a plumbing bug.
"""
import sys
import os
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params
from eval_category_b import solve_llm4branch
import category_k_er_indset as gen
from tune_category_k import solve_one, TRAIN_SEEDS

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)

# Path A: eval_category_b's solve_llm4branch (used throughout calibration)
score_function_a = partial(score_function, params=function_params)
for seed in TRAIN_SEEDS:
    instance = gen.generate_instance(seed=seed, n_nodes=gen.N_NODES_TRAIN, edge_probability=gen.P_TRAIN)
    t, n, s = solve_llm4branch(instance, score_function_a, used_features)
    print(f"[calibration path] seed={seed}: nodes={n} status={s}", flush=True)

# Path B: tune_category_k's solve_one (used inside the BO loop)
scip_params = get_scip_params(time_limit=180)
for seed in TRAIN_SEEDS:
    r = solve_one((seed, used_features, function_params, scip_params))
    print(f"[tuning path]      seed={seed}: nodes={r.get('nodes')} validity={r['validity']}", flush=True)
