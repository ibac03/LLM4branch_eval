"""
Calibrate a genuinely-hard, genuinely-solvable ER test tier at n=1000 --
matched in SIZE to BA-standard (affinity=4, n=1000) so the ER-side
numbers in the Category K comparison are no longer confounded by a
2.5x smaller problem size (n=400 vs n=1000).

ER's difficulty is extremely sensitive to avg degree (at n=400: p=0.010
-> avg_deg~4 solves in ~2s; p=0.020 -> avg_deg~8 times out completely).
At n=1000 the same avg degree will likely be even harder (bigger
problem), so this sweeps LOWER densities than were used at n=400,
always checking status="optimal" explicitly (no repeat of the earlier
timeout-mistaken-for-difficulty mistake).
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
    ("ER n=1000, p=0.002 (avg_deg~2)", 1000, 0.002),
    ("ER n=1000, p=0.003 (avg_deg~3)", 1000, 0.003),
    ("ER n=1000, p=0.004 (avg_deg~4)", 1000, 0.004),
    ("ER n=1000, p=0.005 (avg_deg~5)", 1000, 0.005),
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
