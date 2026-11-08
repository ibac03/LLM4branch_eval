"""
Category A (intensified): same mechanism as category_a_generator.py --
heavy-tailed column degree + bottleneck rows -- but pushed to a much more
extreme regime before concluding the "degree distribution" structural
family is a dead end for exposing LLM4Branch brittleness:

  - HUB_FRACTION: 0.05 -> 0.02   (hubs are rarer, so their columns must
    carry much more relative "weight" per hub to matter at all)
  - Pareto shape a: 1.5 -> 1.1   (heavier tail: a handful of columns get
    truly enormous degree, most columns are near-singleton)
  - BOTTLENECK_FRACTION: 0.10 -> 0.20 (twice as many rows are decoupled
    from hub columns entirely, forcing more of the search to route
    through the "niche only" mechanism)

Same n_rows/n_cols scale as before (1000x1000, matching medium tier).
"""
import numpy as np
import pyscipopt

N_ROWS = 1000
N_COLS = 1000
BOTTLENECK_FRACTION = 0.20
HUB_FRACTION = 0.02
PARETO_SHAPE = 1.1
COST_LOW, COST_HIGH = 1.0, 10.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS):
    rng = np.random.RandomState(seed)

    bottleneck_rows = set(rng.choice(n_rows, size=int(n_rows * BOTTLENECK_FRACTION), replace=False))
    normal_rows = [i for i in range(n_rows) if i not in bottleneck_rows]

    # heavy-tailed column degrees (Pareto-shaped, steep tail), capped to a sane range
    raw = rng.pareto(a=PARETO_SHAPE, size=n_cols)
    degrees = np.clip((raw * 8).astype(int) + 1, 1, min(400, n_rows))

    hub_threshold = np.percentile(degrees, 100 * (1 - HUB_FRACTION))
    hub_cols = [j for j in range(n_cols) if degrees[j] >= hub_threshold]
    niche_cols = [j for j in range(n_cols) if degrees[j] < hub_threshold]

    coverage = [set() for _ in range(n_rows)]  # row -> set of covering columns

    # hub columns cover only normal rows
    for j in hub_cols:
        d = min(degrees[j], len(normal_rows))
        for i in rng.choice(normal_rows, size=d, replace=False):
            coverage[i].add(j)

    # guarantee every bottleneck row is covered by >=2 niche columns (feasibility)
    for i in bottleneck_rows:
        chosen = rng.choice(niche_cols, size=min(2, len(niche_cols)), replace=False)
        for j in chosen:
            coverage[i].add(j)

    # niche columns fill remaining degree budget among normal+bottleneck rows
    all_rows = np.arange(n_rows)
    for j in niche_cols:
        already = sum(1 for i in bottleneck_rows if j in coverage[i])
        remaining = max(0, degrees[j] - already)
        if remaining == 0:
            continue
        candidates = rng.choice(all_rows, size=min(remaining, n_rows), replace=False)
        for i in candidates:
            coverage[i].add(j)

    # feasibility repair: any row still uncovered gets a random column
    for i in range(n_rows):
        if not coverage[i]:
            j = rng.randint(n_cols)
            coverage[i].add(j)

    costs = rng.uniform(COST_LOW, COST_HIGH, size=n_cols)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()

    stats = {
        "n_rows": n_rows, "n_cols": n_cols,
        "n_bottleneck_rows": len(bottleneck_rows),
        "n_hub_cols": len(hub_cols), "n_niche_cols": len(niche_cols),
        "degree_min": int(degrees.min()), "degree_max": int(degrees.max()),
        "degree_mean": float(degrees.mean()),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
