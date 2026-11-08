"""
Quick status check: were the "hard" affinity=8 / ER-p=0.008 results in
calibrate_k.py / calibrate_k2_graphtype.py genuinely solved, or did they
hit the 120s TIME_LIMIT hard cutoff? Their near-constant ~124-126s times
across all seeds were suspicious. This prints the SCIP status explicitly.
"""
import sys
import os
from functools import partial

import ecole

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program
from eval_category_b import solve_llm4branch, TIME_LIMIT

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "indset", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
score_function = partial(score_function, params=function_params)

print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)

BA = ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert
ER = ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi


def make_ba(seed, n_nodes, affinity):
    g = ecole.instance.IndependentSetGenerator(n_nodes=n_nodes, graph_type=BA, affinity=affinity)
    g.seed(seed)
    return next(g)


def make_er(seed, n_nodes, p):
    g = ecole.instance.IndependentSetGenerator(n_nodes=n_nodes, graph_type=ER, edge_probability=p)
    g.seed(seed)
    return next(g)


configs = [
    ("BA affinity=8, seed=0", lambda: make_ba(0, 1000, 8)),
    ("ER p=0.008, seed=0", lambda: make_er(0, 1000, 0.008)),
]
for name, fn in configs:
    instance = fn()
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    print(f"{name}: time={t:.2f} nodes={n} status={s}", flush=True)
