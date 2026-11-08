"""
Category B2: clustered structure + decoy bridge columns.

Same base mechanism as category_b_generator.py (rows/cols partitioned into
N_CLUSTERS aligned modules, home columns confined to their own cluster,
bridge columns couple clusters together). The addition here targets
LLM4Branch's specific blind spot rather than generic difficulty:

RPB's relpscost actually re-solves child LPs to see each candidate's true
downstream value (expensive, but exact). LLM4Branch has no lookahead --
it scores candidates from static per-node features (fractional part,
pseudocost, obj coefficient, active-constraint-count) via a fixed formula.
So the lever here is to make some candidates *look* good under those
static features while a real lookahead procedure would see they aren't:

  - A handful of "bottleneck" rows per cluster boundary can ONLY be
    covered by a small set of "genuine" bridge columns (higher cost) --
    mirroring Category A's bottleneck-row mechanic, but now placed at
    cluster boundaries instead of scattered globally.
  - Most bridge columns are "decoy" bridges instead: deliberately cheap
    (so they look attractive by cost, and are more likely to sit
    fractional at the LP relaxation -- inflating LLM4Branch's fractional
    part / obj_fraction_synergy features), but their coverage is
    redundant with existing home-column coverage -- they don't actually
    help close any bottleneck.

A policy with no lookahead has no way to tell a cheap-but-redundant decoy
apart from a genuinely useful column except by actually trying it; RPB's
strong-branching-style lookahead can.
"""
import numpy as np
import pyscipopt

N_ROWS = 500
N_COLS = 500
N_CLUSTERS = 4
LOCAL_DENSITY = 0.08       # unchanged from the calibrated Category B setting
BRIDGE_FRACTION = 0.15     # unchanged -- total fraction of columns that are bridges
GENUINE_BRIDGE_SHARE = 0.3  # of the bridge columns, this fraction are "genuine"
                            # (necessary, expensive); the rest are decoys (cheap, redundant)
BOTTLENECK_FRACTION = 0.04  # fraction of rows per cluster boundary reachable ONLY by
                            # genuine bridge columns
DECOY_COST_LOW, DECOY_COST_HIGH = 1.0, 2.0     # decoys: deliberately cheap
GENUINE_COST_LOW, GENUINE_COST_HIGH = 8.0, 10.0  # genuine bridges: deliberately expensive
HOME_COST_LOW, HOME_COST_HIGH = 1.0, 10.0        # home columns: unchanged, normal range

EASY_N_ROWS = 400
EASY_N_COLS = 400


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, n_clusters=N_CLUSTERS):
    rng = np.random.RandomState(seed)

    row_cluster = rng.randint(0, n_clusters, size=n_rows)
    col_cluster = rng.randint(0, n_clusters, size=n_cols)
    rows_by_cluster = [np.where(row_cluster == k)[0] for k in range(n_clusters)]
    all_rows = np.arange(n_rows)

    n_bridge = int(n_cols * BRIDGE_FRACTION)
    bridge_cols = list(rng.choice(n_cols, size=n_bridge, replace=False))
    rng.shuffle(bridge_cols)
    n_genuine = max(1, int(n_bridge * GENUINE_BRIDGE_SHARE))
    genuine_bridge_cols = set(bridge_cols[:n_genuine])
    decoy_bridge_cols = set(bridge_cols[n_genuine:])

    # bottleneck rows: a small set of rows only genuine bridges are allowed to cover
    bottleneck_rows = set(rng.choice(n_rows, size=int(n_rows * BOTTLENECK_FRACTION), replace=False))
    normal_rows = np.array([i for i in range(n_rows) if i not in bottleneck_rows])

    coverage = [set() for _ in range(n_rows)]

    # home columns: confined to their own cluster's NON-bottleneck rows
    for j in range(n_cols):
        if j in genuine_bridge_cols or j in decoy_bridge_cols:
            continue
        home_rows = np.array([i for i in rows_by_cluster[col_cluster[j]] if i not in bottleneck_rows])
        if len(home_rows) == 0:
            continue
        target_degree = max(1, int(LOCAL_DENSITY * len(home_rows)))
        degree = min(rng.poisson(target_degree) + 1, len(home_rows))
        chosen = rng.choice(home_rows, size=degree, replace=False)
        for i in chosen:
            coverage[i].add(j)

    # genuine bridge columns: guarantee coverage of the bottleneck rows (>=2 genuine
    # bridges per bottleneck row so there's real choice, not a forced unique column),
    # plus a modest amount of ordinary cross-cluster coverage
    for i in bottleneck_rows:
        chosen = rng.choice(list(genuine_bridge_cols), size=min(2, len(genuine_bridge_cols)), replace=False)
        for j in chosen:
            coverage[i].add(j)
    for j in genuine_bridge_cols:
        target_degree = max(1, int(0.02 * n_rows))
        degree = min(rng.poisson(target_degree) + 1, len(normal_rows))
        chosen = rng.choice(normal_rows, size=degree, replace=False)
        for i in chosen:
            coverage[i].add(j)

    # decoy bridge columns: cheap, but coverage is confined to normal (non-bottleneck)
    # rows already reachable via home columns -- redundant, not decisive
    for j in decoy_bridge_cols:
        target_degree = max(1, int(0.02 * n_rows))
        degree = min(rng.poisson(target_degree) + 1, len(normal_rows))
        chosen = rng.choice(normal_rows, size=degree, replace=False)
        for i in chosen:
            coverage[i].add(j)

    # feasibility repair: any row still uncovered gets a random *legal* column
    # (bottleneck rows can only be repaired with a genuine bridge column)
    for i in range(n_rows):
        if not coverage[i]:
            if i in bottleneck_rows:
                j = rng.choice(list(genuine_bridge_cols))
            else:
                j = rng.randint(n_cols)
            coverage[i].add(j)

    costs = np.zeros(n_cols)
    for j in range(n_cols):
        if j in genuine_bridge_cols:
            costs[j] = rng.uniform(GENUINE_COST_LOW, GENUINE_COST_HIGH)
        elif j in decoy_bridge_cols:
            costs[j] = rng.uniform(DECOY_COST_LOW, DECOY_COST_HIGH)
        else:
            costs[j] = rng.uniform(HOME_COST_LOW, HOME_COST_HIGH)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()

    stats = {
        "n_rows": n_rows, "n_cols": n_cols, "n_clusters": n_clusters,
        "n_genuine_bridges": len(genuine_bridge_cols), "n_decoy_bridges": len(decoy_bridge_cols),
        "n_bottleneck_rows": len(bottleneck_rows),
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
