"""
BA vs ER at a tractable size (n=400), with status always checked (the
n=1000 attempt silently hit TIME_LIMIT=120 on both and was meaningless).

Rationale: BA's degree distribution is power-law with exponent ~-3
REGARDLESS of the affinity parameter (only avg degree ~2*affinity
scales) -- so varying affinity alone is mostly a density/difficulty
knob on the SAME topology shape, not a genuine structural regime
change. This mirrors Category A in the SetCover work (raw degree
heterogeneity present but same qualitative structure), which did NOT
produce a retuning trade-off on its own.

ER's degree distribution is flat/Poisson -- no natural hubs at all.
LLM4Branch's IndSet policy positively weights constraint_count (=node
degree), betting on hub nodes existing. ER removes that assumption's
target entirely: a policy retuned there would need to strip out
degree-based preference altogether, which is the kind of qualitative
recompute that produced a genuine trade-off in Category J (SetCover).

edge_probability matched to BA affinity=4's avg degree (~2*4=8) at
n=400: p = 8/399 ~= 0.02. Also test a bit sparser/denser for context.
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
ER = ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi

print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)


def make_ba(seed, n_nodes, affinity):
    g = ecole.instance.IndependentSetGenerator(n_nodes=n_nodes, graph_type=BA, affinity=affinity)
    g.seed(seed)
    return next(g)


def make_er(seed, n_nodes, p):
    g = ecole.instance.IndependentSetGenerator(n_nodes=n_nodes, graph_type=ER, edge_probability=p)
    g.seed(seed)
    return next(g)


configs = [
    ("BA affinity=4 (baseline, avg_deg~8), n=400", lambda seed: make_ba(seed, 400, 4)),
    ("ER p=0.020 (matched avg_deg~8), n=400", lambda seed: make_er(seed, 400, 0.020)),
    ("ER p=0.010 (sparser, avg_deg~4), n=400", lambda seed: make_er(seed, 400, 0.010)),
    ("ER p=0.040 (denser, avg_deg~16), n=400", lambda seed: make_er(seed, 400, 0.040)),
]

for name, make_fn in configs:
    times, nodes_list, statuses = [], [], []
    for seed in range(4):
        instance = make_fn(seed)
        t, n, s = solve_llm4branch(instance, score_function, used_features)
        times.append(t)
        nodes_list.append(n)
        statuses.append(s)
    all_optimal = all(s == "optimal" for s in statuses)
    print(f"{name}: nodes={nodes_list} time={[round(t,2) for t in times]} statuses={statuses} "
          f"all_optimal={all_optimal} geomean_time={geometric_mean_stable(times):.2f} "
          f"mean_nodes={np.mean(nodes_list):.1f}", flush=True)
