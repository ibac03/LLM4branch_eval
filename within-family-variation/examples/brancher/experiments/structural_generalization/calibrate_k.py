"""
Calibrate Category K (IndSet affinity variants): check node counts with
the Original released IndSet policy across several affinity levels, at
both the easy (750) and medium (1000) tiers, before committing to a
full tune+crosstest cycle.
"""
import sys
import os
from functools import partial

import numpy as np
import ecole

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch
from category_k_indset_affinity import generate_instance

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
score_function = partial(score_function, params=function_params)

configs = [
    ("affinity=4 (standard), n=1000 (medium)", 1000, 4),
    ("affinity=8, n=1000 (medium)", 1000, 8),
    ("affinity=12, n=1000 (medium)", 1000, 12),
    ("affinity=16, n=1000 (medium)", 1000, 16),
    ("affinity=12, n=750 (easy/train)", 750, 12),
    ("affinity=4 (standard), n=750 (easy)", 750, 4),
]

for name, n_nodes, affinity in configs:
    times, nodes_list = [], []
    for seed in range(4):
        instance = generate_instance(seed=seed, n_nodes=n_nodes, affinity=affinity)
        t, n, s = solve_llm4branch(instance, score_function, used_features)
        times.append(t)
        nodes_list.append(n)
    print(f"{name}: nodes={nodes_list} time={[round(t,2) for t in times]} "
          f"geomean_time={geometric_mean_stable(times):.2f} mean_nodes={np.mean(nodes_list):.1f}", flush=True)
