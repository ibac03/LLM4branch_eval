"""
Calibrate Category K2: same Maximum Independent Set problem, different
underlying graph topology -- Barabasi-Albert (LLM4Branch's own choice,
scale-free/hub-heavy) vs Erdos-Renyi (uniform random, no hub structure)
-- using ecole's built-in IndependentSetGenerator graph_type switch, no
custom generator code needed.

LLM4Branch's released IndSet policy positively weights constraint_count
(=node degree). BA graphs have a few very-high-degree hubs (rich-get-
richer); ER graphs have a much flatter, more uniform degree distribution
around n*p. A policy tuned to exploit BA's hub structure may not transfer
to ER's flat-degree regime, and vice versa -- this tests a genuine
topology shift, not just a density knob.

edge_probability for ER is picked so the two graphs have comparable
overall edge density/avg-degree to BA at the same n_nodes/affinity, for
a fairer comparison (avg degree in BA with affinity m is ~2m).
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

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
score_function = partial(score_function, params=function_params)


def make_instance(seed, n_nodes, graph_type, affinity=4, edge_probability=0.008):
    kwargs = {"n_nodes": n_nodes, "graph_type": graph_type}
    if graph_type == ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert:
        kwargs["affinity"] = affinity
    else:
        kwargs["edge_probability"] = edge_probability
    generator = ecole.instance.IndependentSetGenerator(**kwargs)
    generator.seed(seed)
    return next(generator)


BA = ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert
ER = ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi

# avg degree in BA(n, affinity=m) ~ 2m; matching ER(n, p) avg degree ~ n*p
# n=1000, affinity=4 -> avg degree ~8 -> p ~ 8/999 ~ 0.008
configs = [
    ("BA affinity=4 (LLM4Branch standard), n=1000", BA, 1000, 4, None),
    ("ER p=0.008 (~matched avg degree), n=1000", ER, 1000, None, 0.008),
    ("ER p=0.004 (sparser), n=1000", ER, 1000, None, 0.004),
    ("ER p=0.016 (denser), n=1000", ER, 1000, None, 0.016),
]

for name, gtype, n_nodes, affinity, p in configs:
    times, nodes_list = [], []
    for seed in range(4):
        instance = make_instance(seed, n_nodes, gtype, affinity=affinity or 4, edge_probability=p or 0.008)
        t, n, s = solve_llm4branch(instance, score_function, used_features)
        times.append(t)
        nodes_list.append(n)
    print(f"{name}: nodes={nodes_list} time={[round(t,2) for t in times]} "
          f"geomean_time={geometric_mean_stable(times):.2f} mean_nodes={np.mean(nodes_list):.1f}", flush=True)
