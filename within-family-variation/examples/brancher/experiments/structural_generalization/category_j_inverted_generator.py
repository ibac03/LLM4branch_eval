"""
Category J: inverted constraint-importance trap.

Our own coordinate-sweep analysis (much earlier this session) established
that params[2] (pseudocost_strength) and params[6] (constraint_importance)
are the DOMINANT, load-bearing features in LLM4Branch's released
SetCover policy -- param[6] alone captured 92.5% of its margin over RPB.
constraint_importance = log1p(active_constraint_count)/log(2), always
POSITIVELY weighted -- the policy is built to treat "touches many
constraints" as a generically good sign.

This category is a direct trap for that assumption: it inverts Category
A's hub/niche degree-heterogeneity mechanic so that HIGH-degree ("hub")
columns are deliberately BAD choices -- expensive, and their coverage is
fully redundant with what low-degree ("niche") columns already provide --
while niche columns are cheap AND uniquely necessary for a subset of
"bottleneck" rows only they can cover. Any policy that rewards high
active-constraint-count (as the released policy's param[6] does) will be
biased toward exactly the columns that are actually wasteful here.

The point isn't just "does the original policy do badly here" (Category A
already showed raw degree-heterogeneity alone doesn't break it) -- it's
whether a policy RE-TUNED on this category is forced to suppress
param[6] to perform well here, and whether that suppression then measurably
hurts the SAME policy's performance back on standard SetCover (where
param[6] is genuinely load-bearing). That's the real test.
"""
import numpy as np
import pyscipopt

N_ROWS = 1000
N_COLS = 1000
EASY_N_ROWS = 400
EASY_N_COLS = 400
BOTTLENECK_FRACTION = 0.15   # fraction of rows only niche columns can cover
HUB_FRACTION = 0.05          # top fraction of columns by degree, deliberately made bad
HUB_COST_LOW, HUB_COST_HIGH = 8.0, 10.0     # hub columns: expensive
NICHE_COST_LOW, NICHE_COST_HIGH = 1.0, 3.0  # niche columns: cheap


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS):
    rng = np.random.RandomState(seed)

    bottleneck_rows = set(rng.choice(n_rows, size=int(n_rows * BOTTLENECK_FRACTION), replace=False))
    normal_rows = np.array([i for i in range(n_rows) if i not in bottleneck_rows])

    raw = rng.pareto(a=1.5, size=n_cols)
    degrees = np.clip((raw * 8).astype(int) + 1, 1, min(200, n_rows))
    hub_threshold = np.percentile(degrees, 100 * (1 - HUB_FRACTION))
    hub_cols = [j for j in range(n_cols) if degrees[j] >= hub_threshold]
    niche_cols = [j for j in range(n_cols) if degrees[j] < hub_threshold]

    coverage = [set() for _ in range(n_rows)]

    # niche columns first: cover a mix of normal rows, PLUS guarantee every
    # bottleneck row gets >=2 niche columns (the only columns allowed to
    # touch bottleneck rows at all)
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

    # hub columns: cover ONLY normal rows, and only rows that already have
    # decent niche coverage (fully redundant -- no unique value at all)
    normal_degree = np.array([len(coverage[i]) for i in normal_rows])
    well_covered_normal = normal_rows[normal_degree >= 1] if (normal_degree >= 1).any() else normal_rows
    for j in hub_cols:
        d = min(degrees[j], len(well_covered_normal))
        if d == 0:
            continue
        chosen = rng.choice(well_covered_normal, size=d, replace=False)
        for i in chosen:
            coverage[i].add(j)

    # feasibility repair: any row still uncovered gets a cheap niche column
    for i in range(n_rows):
        if not coverage[i]:
            j = rng.choice(niche_cols)
            coverage[i].add(j)

    costs = np.zeros(n_cols)
    for j in range(n_cols):
        if j in hub_cols:
            costs[j] = rng.uniform(HUB_COST_LOW, HUB_COST_HIGH)
        else:
            costs[j] = rng.uniform(NICHE_COST_LOW, NICHE_COST_HIGH)

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
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
