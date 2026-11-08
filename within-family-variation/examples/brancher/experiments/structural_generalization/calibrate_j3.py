"""
Quick calibration: why did category_j2 (bottleneck=0.30, hub_cost=15-20,
size=600x600) collapse to mean_nodes~2.5 (too easy), when category_j
(bottleneck=0.15, hub_cost=8-10, size=400x400) gave mean_nodes~10-25?

Hypothesis: blowing the hub/niche cost gap up too far (15-20 vs 1-3) makes
the LP relaxation itself trivially avoid hub columns via reduced cost --
no fractional tension left to branch on, regardless of policy. The trap
needs cost pressure that's real but not so extreme it resolves the LP.

Test several (bottleneck_fraction, hub_cost_range, size) combos, solving
ONE seed each with the Original policy, reporting nodes.
"""
import sys
import os
import importlib
import numpy as np
import pyscipopt
import ecole
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, normalize
from eval_category_b import solve_llm4branch

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "setcover", "program.py")
score_function, used_features, function_params, _ = load_program(PROGRAM_PATH)
score_function = partial(score_function, params=function_params)


def make_instance(seed, n_rows, n_cols, bottleneck_fraction, hub_fraction, hub_cost_lo, hub_cost_hi, niche_cost_lo, niche_cost_hi):
    rng = np.random.RandomState(seed)
    bottleneck_rows = set(rng.choice(n_rows, size=int(n_rows * bottleneck_fraction), replace=False))
    normal_rows = np.array([i for i in range(n_rows) if i not in bottleneck_rows])

    raw = rng.pareto(a=1.5, size=n_cols)
    degrees = np.clip((raw * 8).astype(int) + 1, 1, min(200, n_rows))
    hub_threshold = np.percentile(degrees, 100 * (1 - hub_fraction))
    hub_cols = [j for j in range(n_cols) if degrees[j] >= hub_threshold]
    niche_cols = [j for j in range(n_cols) if degrees[j] < hub_threshold]

    coverage = [set() for _ in range(n_rows)]
    for i in bottleneck_rows:
        chosen = rng.choice(niche_cols, size=min(2, len(niche_cols)), replace=False)
        for j in chosen:
            coverage[i].add(j)
    for j in niche_cols:
        already = sum(1 for i in bottleneck_rows if j in coverage[i])
        remaining_degree = max(0, degrees[j] - already)
        if remaining_degree == 0:
            continue
        chosen = rng.choice(normal_rows, size=min(remaining_degree, len(normal_rows)), replace=False)
        for i in chosen:
            coverage[i].add(j)

    normal_degree = np.array([len(coverage[i]) for i in normal_rows])
    well_covered_normal = normal_rows[normal_degree >= 1] if (normal_degree >= 1).any() else normal_rows
    for j in hub_cols:
        d = min(degrees[j], len(well_covered_normal))
        if d == 0:
            continue
        chosen = rng.choice(well_covered_normal, size=d, replace=False)
        for i in chosen:
            coverage[i].add(j)

    for i in range(n_rows):
        if not coverage[i]:
            j = rng.choice(niche_cols)
            coverage[i].add(j)

    costs = np.zeros(n_cols)
    for j in range(n_cols):
        if j in hub_cols:
            costs[j] = rng.uniform(hub_cost_lo, hub_cost_hi)
        else:
            costs[j] = rng.uniform(niche_cost_lo, niche_cost_hi)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()
    return model


configs = [
    ("v1 baseline (400x400, bn=0.15, hub=8-10)", 400, 400, 0.15, 0.05, 8.0, 10.0),
    ("v2 (600x600, bn=0.30, hub=15-20)", 600, 600, 0.30, 0.05, 15.0, 20.0),
    ("v3a (600x600, bn=0.20, hub=10-12)", 600, 600, 0.20, 0.05, 10.0, 12.0),
    ("v3b (600x600, bn=0.15, hub=8-10, size-only)", 600, 600, 0.15, 0.05, 8.0, 10.0),
    ("v3c (600x600, bn=0.25, hub=8-10)", 600, 600, 0.25, 0.05, 8.0, 10.0),
]

for name, n_rows, n_cols, bn, hf, hlo, hhi in configs:
    nodes_list = []
    for seed in range(3):
        model = make_instance(seed, n_rows, n_cols, bn, hf, hlo, hhi, 1.0, 3.0)
        instance = ecole.scip.Model.from_pyscipopt(model)
        t, n, s = solve_llm4branch(instance, score_function, used_features)
        nodes_list.append(n)
    print(f"{name}: nodes={nodes_list} mean={np.mean(nodes_list):.1f}", flush=True)
