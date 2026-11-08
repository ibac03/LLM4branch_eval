"""
Category J3: intensified inverted constraint-importance trap
(calibrated after v2 overshot).

v1 (category_j_inverted_generator.py) confirmed the trade-off DIRECTION
(Tuned-J better on Category J, worse on standard SetCover) but both
effects were weak (-1.4% / +3.8%), likely because the easy (400x400)
tuning tier gave the BO tuner very little signal -- mean_nodes stayed in
a narrow 10-25 band across the whole 50-call search.

v2 (category_j2_inverted_generator.py) tried to fix this by going too
far at once: BOTTLENECK_FRACTION 0.15->0.30 AND HUB_COST 8-10->15-20 AND
size 400->600. Result: mean_nodes COLLAPSED to ~2.5 (even less signal
than v1). Diagnosis via calibrate_j3.py isolated the cause: blowing the
hub/niche cost gap up to 15-20 vs 1-3 let the LP relaxation trivially
avoid hub columns via reduced cost alone -- there's no fractional
tension left to branch on regardless of policy, so the trap actually got
EASIER, not harder. Bottleneck fraction and size increases were each
independently fine (mean_nodes 22-34 in isolation); the cost gap
was the only harmful axis.

This version keeps the beneficial changes and reverts the harmful one:
  - EASY_N_ROWS/EASY_N_COLS: 400 -> 600 (more BO signal, confirmed good
    in isolation: mean_nodes 34.3 with everything else at v1 settings)
  - BOTTLENECK_FRACTION: 0.15 -> 0.20 (mild increase, confirmed good in
    isolation: mean_nodes 22-27 combined with size or cost changes)
  - HUB_COST_LOW/HIGH: 8-10 -> 10-12 (MILD increase only, keeping real
    fractional tension in the LP relaxation -- the v2 mistake was going
    to 15-20)

Medium (evaluation) tier stays at 1000x1000 -- unchanged, so it is
still directly comparable to standard SetCover's own medium tier
(DATASET_PARAMS['setcover']['medium'] = 1000x1000).
"""
import numpy as np
import pyscipopt

N_ROWS = 1000
N_COLS = 1000
EASY_N_ROWS = 600
EASY_N_COLS = 600
BOTTLENECK_FRACTION = 0.20   # fraction of rows only niche columns can cover
HUB_FRACTION = 0.05          # top fraction of columns by degree, deliberately made bad
HUB_COST_LOW, HUB_COST_HIGH = 10.0, 12.0    # hub columns: much more expensive
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
