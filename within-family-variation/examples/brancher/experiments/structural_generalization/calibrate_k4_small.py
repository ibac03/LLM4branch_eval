"""
Corrected Category K calibration. Previous attempts at n=1000 were
useless above affinity=4: BA affinity=8 and ER p=0.008 both silently hit
eval_category_b.py's hardcoded TIME_LIMIT=120s (status=timelimit, not
optimal) -- their near-identical ~125s times were a timeout artifact,
not real difficulty data. This version:
  - uses a SMALLER n_nodes (400) so a genuine optimal solve is feasible
    within a reasonable budget
  - ALWAYS prints status, so a timeout can never again be silently
    mistaken for a real solve time
  - sweeps BA affinity in {4, 5, 6, 8} at n=400 to find a level that is
    meaningfully harder than the affinity=4 baseline AND still reaches
    status=optimal in well under a minute (needed for a tractable 50-call
    x 4-seed BO tuning loop later)
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

BA = ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert

print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)

configs = [
    ("BA affinity=4, n=400", 400, 4),
    ("BA affinity=5, n=400", 400, 5),
    ("BA affinity=6, n=400", 400, 6),
    ("BA affinity=8, n=400", 400, 8),
]

for name, n_nodes, affinity in configs:
    times, nodes_list, statuses = [], [], []
    for seed in range(4):
        g = ecole.instance.IndependentSetGenerator(n_nodes=n_nodes, graph_type=BA, affinity=affinity)
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
