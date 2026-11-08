"""
Category A: heavy-tailed column degree + bottleneck rows.

Standard SetCoverGenerator instances are structurally homogeneous: every
column covers roughly the same number of rows (i.i.d. density=0.05), so a
"more constraints touched = more important" heuristic (LLM4Branch's
constraint_importance term) is a reasonable, if crude, proxy on this family.

This generator breaks that assumption deliberately:
  - Most columns are "niche" (small coverage), a few are "hub" columns
    (large coverage) -- a heavy-tailed/power-law-like degree distribution.
  - A subset of rows are "bottleneck" rows: only niche columns are allowed
    to cover them, hub columns never touch them. So the genuinely difficult
    part of the instance (which niche column to use for each bottleneck
    row) is structurally decoupled from column degree -- a policy that
    treats high active_constraint_count as "important" would over-weight
    hub columns, which are actually easy/low-value choices here.

Same n_rows/n_cols scale as the standard "medium" setcover tier, so any
observed difference is attributable to structure, not size.

Usage:
    from category_a_generator import generate_instance
    model = generate_instance(seed=0)
"""
import numpy as np
import pyscipopt

N_ROWS = 1000
N_COLS = 1000
BOTTLENECK_FRACTION = 0.10   # 10% of rows are bottleneck rows
HUB_FRACTION = 0.05          # top 5% of columns by degree are "hub" columns
COST_LOW, COST_HIGH = 1.0, 10.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS):
    rng = np.random.RandomState(seed)

    bottleneck_rows = set(rng.choice(n_rows, size=int(n_rows * BOTTLENECK_FRACTION), replace=False))
    normal_rows = [i for i in range(n_rows) if i not in bottleneck_rows]

    # heavy-tailed column degrees (Pareto-shaped), capped to a sane range
    raw = rng.pareto(a=1.5, size=n_cols)
    degrees = np.clip((raw * 8).astype(int) + 1, 1, min(200, n_rows))

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
