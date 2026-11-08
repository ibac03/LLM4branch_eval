"""
Category B3: calibrated Category B (clustered/modular) + surgical decoy bridges.

B2's first decoy attempt (category_b_decoy_generator.py) failed because
excluding bottleneck rows from home-column reach *reduced* overall
density, making the whole instance much easier than the calibrated B
baseline (RPB 0/10 wins vs baseline's 28%) -- the decoy mechanism was
never actually tested under real difficulty.

This version starts from the EXACT calibrated Category B generation
process (same n_rows/n_cols/n_clusters/local_density/bridge params, so
the base difficulty is preserved) and only POST-PROCESSES the bridge
columns already produced:
  - A small subset of bottleneck rows (chosen from rows that already have
    relatively sparse coverage) have their existing coverage stripped and
    replaced with mandatory coverage by 2 "genuine" bridge columns
    (expensive, cost near COST_HIGH).
  - The remaining bridge columns become "decoy" bridges: their cost is
    pulled down to be deliberately cheap (near COST_LOW), while their
    coverage (already generated, spanning random rows) is left
    untouched -- so they still contribute real structural coupling, just
    now they also look artificially attractive by cost.

This preserves the calibrated instance's base difficulty while adding the
decoy/genuine cost-asymmetry mechanism on top, rather than replacing it.
"""
import numpy as np
import pyscipopt

import category_b_generator as base

N_ROWS = base.N_ROWS
N_COLS = base.N_COLS
N_CLUSTERS = base.N_CLUSTERS
BOTTLENECK_FRACTION = 0.04       # fraction of rows converted to bottleneck rows
GENUINE_COST_LOW, GENUINE_COST_HIGH = 8.0, 10.0
DECOY_COST_LOW, DECOY_COST_HIGH = 1.0, 2.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, n_clusters=N_CLUSTERS):
    rng = np.random.RandomState(seed)

    row_cluster = rng.randint(0, n_clusters, size=n_rows)
    col_cluster = rng.randint(0, n_clusters, size=n_cols)
    rows_by_cluster = [np.where(row_cluster == k)[0] for k in range(n_clusters)]
    all_rows = np.arange(n_rows)

    n_bridge = int(n_cols * base.BRIDGE_FRACTION)
    bridge_cols = list(rng.choice(n_cols, size=n_bridge, replace=False))

    coverage = [set() for _ in range(n_rows)]

    # identical generation to the calibrated Category B (density/degree unchanged)
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

    for i in range(n_rows):
        if not coverage[i]:
            j = rng.randint(n_cols)
            coverage[i].add(j)

    # split bridges into genuine (small, expensive) vs decoy (rest, cheap)
    rng.shuffle(bridge_cols)
    n_genuine = max(2, int(len(bridge_cols) * 0.3))
    genuine_bridges = bridge_cols[:n_genuine]
    decoy_bridges = bridge_cols[n_genuine:]

    # pick bottleneck rows from among the currently most sparsely-covered rows
    # (not the least covered -- avoid picking already-critical rows -- just a
    # deliberately chosen "boundary" subset), strip their coverage, force onto
    # 2 genuine bridges each
    degree_per_row = np.array([len(c) for c in coverage])
    candidate_rows = np.argsort(degree_per_row)[: int(n_rows * BOTTLENECK_FRACTION * 3)]
    bottleneck_rows = rng.choice(candidate_rows, size=int(n_rows * BOTTLENECK_FRACTION), replace=False)

    for i in bottleneck_rows:
        coverage[i] = set()
        chosen = rng.choice(genuine_bridges, size=min(2, len(genuine_bridges)), replace=False)
        for j in chosen:
            coverage[i].add(j)

    costs = rng.uniform(base.COST_LOW, base.COST_HIGH, size=n_cols)
    for j in genuine_bridges:
        costs[j] = rng.uniform(GENUINE_COST_LOW, GENUINE_COST_HIGH)
    for j in decoy_bridges:
        costs[j] = rng.uniform(DECOY_COST_LOW, DECOY_COST_HIGH)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()

    stats = {
        "n_rows": n_rows, "n_cols": n_cols, "n_clusters": n_clusters,
        "n_genuine_bridges": len(genuine_bridges), "n_decoy_bridges": len(decoy_bridges),
        "n_bottleneck_rows": len(bottleneck_rows),
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
