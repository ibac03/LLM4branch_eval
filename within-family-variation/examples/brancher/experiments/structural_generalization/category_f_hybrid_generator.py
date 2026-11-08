"""
Category F: hybrid -- clustering (Category B) + heavy-tailed column degree
within each cluster (Category A).

Category A alone (heavy-tailed degree + bottleneck rows, no clustering) gave
LLM4Branch a clean 0% RPB win rate -- no effect. Category B alone
(clustering, uniform Poisson-ish degree per cluster) gave the strongest
result so far (28% RPB wins). This generator stacks both mechanisms: same
calibrated clustering/bridge structure as Category B, but each cluster's
home-column degree distribution is now Pareto-shaped (hub vs niche columns
*within* the cluster) instead of a narrow Poisson band -- so the local
symmetric-block difficulty that made B work is combined with the
within-block heterogeneity that (alone) did nothing, testing whether they
compound rather than being independent/redundant.

Same n_rows/n_cols/n_clusters/bridge settings as the calibrated Category B
(500x500, n_clusters=4, bridge_fraction=0.15, bridge_density=0.015).
"""
import numpy as np
import pyscipopt

N_ROWS = 500
N_COLS = 500
N_CLUSTERS = 4
BRIDGE_FRACTION = 0.15
BRIDGE_DENSITY = 0.015
LOCAL_DENSITY_MEAN = 0.08   # target *average* local density per cluster, matched to
                            # the calibrated Category B setting -- but individual
                            # home columns' degree now follows a Pareto shape
                            # around this mean instead of a narrow Poisson band
PARETO_SHAPE = 1.5
COST_LOW, COST_HIGH = 1.0, 10.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, n_clusters=N_CLUSTERS):
    rng = np.random.RandomState(seed)

    row_cluster = rng.randint(0, n_clusters, size=n_rows)
    col_cluster = rng.randint(0, n_clusters, size=n_cols)
    rows_by_cluster = [np.where(row_cluster == k)[0] for k in range(n_clusters)]
    all_rows = np.arange(n_rows)

    n_bridge = int(n_cols * BRIDGE_FRACTION)
    bridge_cols = set(rng.choice(n_cols, size=n_bridge, replace=False))

    coverage = [set() for _ in range(n_rows)]

    for j in range(n_cols):
        if j in bridge_cols:
            target_degree = max(1, int(BRIDGE_DENSITY * n_rows))
            degree = min(rng.poisson(target_degree) + 1, n_rows)
            chosen = rng.choice(all_rows, size=min(degree, n_rows), replace=False)
        else:
            home_rows = rows_by_cluster[col_cluster[j]]
            if len(home_rows) == 0:
                continue
            # Pareto-shaped degree instead of Poisson: mean matched to
            # LOCAL_DENSITY_MEAN * cluster_size, but with a heavy tail --
            # most home columns are "niche" (small degree), a few are
            # "hub" columns within their own cluster (large degree)
            mean_degree = max(1, LOCAL_DENSITY_MEAN * len(home_rows))
            raw = rng.pareto(a=PARETO_SHAPE)
            degree = max(1, min(len(home_rows), int(mean_degree * (1 + raw) / 2)))
            chosen = rng.choice(home_rows, size=degree, replace=False)
        for i in chosen:
            coverage[i].add(j)

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
        "n_rows": n_rows, "n_cols": n_cols, "n_clusters": n_clusters,
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
