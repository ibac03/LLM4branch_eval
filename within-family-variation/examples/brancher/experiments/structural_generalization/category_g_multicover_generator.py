"""
Category G: calibrated Category B (clustered/modular) + multi-cover
requirements (RHS >= 2 for a subset of rows).

Untried mechanism: standard set cover (and every category so far) requires
each row to be covered by >=1 selected column. This generator keeps the
calibrated Category B clustering/bridge structure exactly as-is, but a
subset of rows now require >=2 covering columns instead of >=1. This
changes what "this row is satisfied" means at every branching node in a
way none of A/B/C/E/F touch -- a row needing 2 columns interacts with
LP-relaxation fractionality very differently (e.g. two columns each at
x=0.5 can satisfy a >=1 row's LP relaxation trivially at cost 1, but a
>=2 row genuinely needs both, so fractional relaxations of >=2 rows are
less "free" and the LP bound is tighter/more informative in some places,
looser in others depending on structure).

Same n_rows/n_cols/n_clusters/local_density/bridge settings as the
calibrated Category B (500x500, n_clusters=4, local_density=0.08,
bridge_fraction=0.15, bridge_density=0.015).
"""
import numpy as np
import pyscipopt

import category_b_generator as base

N_ROWS = base.N_ROWS
N_COLS = base.N_COLS
N_CLUSTERS = base.N_CLUSTERS

# "Easy" tier for the evolutionary loop: same 400x400 scale already calibrated
# for Category B's easy tier (fast solves, real branching still occurs), with
# the same multicover mechanism layered on top.
EASY_N_ROWS = base.EASY_N_ROWS
EASY_N_COLS = base.EASY_N_COLS

MULTICOVER_FRACTION = 0.8  # fraction of rows requiring >=REQUIRED_COVERAGE columns
                           # (0.3->20%, 0.45->37.5%, 0.6->36% at n=25, 0.7->12.5%,
                           # 0.8->50% RPB wins in an 8-seed probe -- noisy vs fraction,
                           # but 0.8 is the first setting to touch 50%; confirming at scale)
REQUIRED_COVERAGE = 2      # how many columns each multicover row needs
                           # (3 gave only 25% RPB wins in an 8-seed probe, worse than
                           # 2's 36% -- same moderate-sweet-spot pattern as every
                           # other lever tried; 2 is the calibrated, final setting)


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, n_clusters=N_CLUSTERS):
    rng = np.random.RandomState(seed)

    row_cluster = rng.randint(0, n_clusters, size=n_rows)
    col_cluster = rng.randint(0, n_clusters, size=n_cols)
    rows_by_cluster = [np.where(row_cluster == k)[0] for k in range(n_clusters)]
    all_rows = np.arange(n_rows)

    n_bridge = int(n_cols * base.BRIDGE_FRACTION)
    bridge_cols = set(rng.choice(n_cols, size=n_bridge, replace=False))

    coverage = [set() for _ in range(n_rows)]

    for j in range(n_cols):
        if j in bridge_cols:
            target_degree = max(1, int(base.BRIDGE_DENSITY * n_rows))
            degree = min(rng.poisson(target_degree) + 1, n_rows)
            chosen = rng.choice(all_rows, size=min(degree, n_rows), replace=False)
        else:
            home_rows = rows_by_cluster[col_cluster[j]]
            if len(home_rows) == 0:
                continue
            target_degree = max(1, int(base.LOCAL_DENSITY * len(home_rows)))
            degree = min(rng.poisson(target_degree) + 1, len(home_rows))
            chosen = rng.choice(home_rows, size=degree, replace=False)
        for i in chosen:
            coverage[i].add(j)

    # multi-cover rows need >=REQUIRED_COVERAGE covering columns -- ensure
    # feasibility by topping up any row (multicover or not) that doesn't have enough
    multicover_rows = set(rng.choice(n_rows, size=int(n_rows * MULTICOVER_FRACTION), replace=False))
    for i in range(n_rows):
        required = REQUIRED_COVERAGE if i in multicover_rows else 1
        while len(coverage[i]) < required:
            coverage[i].add(rng.randint(n_cols))

    costs = rng.uniform(base.COST_LOW, base.COST_HIGH, size=n_cols)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        required = REQUIRED_COVERAGE if i in multicover_rows else 1
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= required)
    model.setMinimize()

    stats = {
        "n_rows": n_rows, "n_cols": n_cols, "n_clusters": n_clusters,
        "n_multicover_rows": len(multicover_rows),
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
