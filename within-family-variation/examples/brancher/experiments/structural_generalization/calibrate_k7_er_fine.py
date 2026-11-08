"""
Fine sweep for the large ER test tier at n=1000. calibrate_k6 showed:
  p=0.003 (avg_deg~3): trivial (1 node, root-solves)
  p=0.004 (avg_deg~4): mostly times out (3/4 seeds hit 120s cap)
The useful window is narrow, between 0.003 and 0.004.
"""
import sys
import os
from functools import partial

import numpy as np
import ecole

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable
from eval_category_b import solve_llm4branch, TIME_LIMIT

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
score_function = partial(score_function, params=function_params)

ER = ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi

print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)

configs = [
    ("ER n=1000, p=0.0032", 1000, 0.0032),
    ("ER n=1000, p=0.0034", 1000, 0.0034),
    ("ER n=1000, p=0.0036", 1000, 0.0036),
    ("ER n=1000, p=0.0038", 1000, 0.0038),
]

for name, n_nodes, p in configs:
    times, nodes_list, statuses = [], [], []
    for seed in range(4):
        g = ecole.instance.IndependentSetGenerator(n_nodes=n_nodes, graph_type=ER, edge_probability=p)
        g.seed(seed)
        instance = next(g)
        t, n, s = solve_llm4branch(instance, score_function, used_features)
        times.append(t)
        nodes_list.append(n)
        statuses.append(s)
    all_optimal = all(s == "optimal" for s in statuses)
    print(f"{name}: nodes={nodes_list} time={[round(t,2) for t in times]} statuses={statuses} "
          f"all_optimal={all_optimal} geomean_time={geometric_mean_stable(times):.2f} "
          f"mean_nodes={np.mean(nodes_list):.1f}", flush=True)
