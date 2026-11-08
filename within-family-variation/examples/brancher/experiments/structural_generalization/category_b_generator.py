"""
Category B: clustered / modular structure.

Standard SetCoverGenerator instances have no community structure at all --
every column's coverage is an i.i.d. random subset of rows (density=0.05),
so the constraint matrix has no meaningful block structure.

This generator partitions rows and columns into N_CLUSTERS aligned
"modules". Most columns ("home" columns) only ever cover rows inside
their own cluster, at a much higher *local* density than the global
density of standard instances -- so the constraint matrix is dominated
by N_CLUSTERS nearly-independent dense blocks. A small fraction of
columns ("bridge" columns) cover rows spanning multiple clusters at
random, which (a) keeps the instance from being trivially separable by
presolve into disconnected subproblems, and (b) is the only place where
cross-cluster tradeoffs actually matter.

The hypothesis this targets: LLM4Branch's policy only ever looks at
per-variable, per-node local features (fractionality, pseudocost,
active-constraint-count, ...) with no notion of "which module am I
currently working on" or "should I finish this cluster before jumping to
another". On instances that decompose into near-independent blocks, a
good branching strategy benefits from *completing* one cluster's
decisions before switching, whereas a purely local-feature policy has no
mechanism to encode that and may branch across clusters in an
inefficient, interleaved order.

Same n_rows/n_cols scale as the standard "medium" setcover tier.
"""
import numpy as np
import pyscipopt

N_ROWS = 500             # scaled down from 1000 -- the clustered/modular mechanism
                         # itself carries far more intrinsic combinatorial hardness
                         # than plain i.i.d. coverage at the same size (every attempt
                         # at 1000x1000 timed out at 120s even at low local density;
                         # 300x300 swung too far the other way, solving at the root
                         # node). 500x500 lands in the same 5-30s solve-time range as
                         # standard medium-tier instances, with no timeouts.
N_COLS = 500
N_CLUSTERS = 4

# "Easy" tier for the evolutionary loop: identical parameters to the medium
# setting above (n_clusters, local density, bridge density/fraction) -- only
# n_rows/n_cols shrink. Swept 450->200; 400 keeps solves mostly well under a
# few seconds (fast enough for thousands of repeated evaluations) while still
# reliably triggering real branching (not root-node-trivial like <=300 mostly is).
EASY_N_ROWS = 400
EASY_N_COLS = 400
BRIDGE_FRACTION = 0.15   # cross-cluster coupling, to avoid trivial presolve separability
LOCAL_DENSITY = 0.08     # home columns' coverage density *within their own cluster*
BRIDGE_DENSITY = 0.015   # bridge columns' coverage density over *all* rows -- light,
                         # scales with n_rows instead of a fixed absolute degree
COST_LOW, COST_HIGH = 1.0, 10.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, n_clusters=N_CLUSTERS):
    rng = np.random.RandomState(seed)

    row_cluster = rng.randint(0, n_clusters, size=n_rows)
    col_cluster = rng.randint(0, n_clusters, size=n_cols)
    rows_by_cluster = [np.where(row_cluster == k)[0] for k in range(n_clusters)]

    n_bridge = int(n_cols * BRIDGE_FRACTION)
    bridge_cols = set(rng.choice(n_cols, size=n_bridge, replace=False))

    coverage = [set() for _ in range(n_rows)]  # row -> set of covering columns
    all_rows = np.arange(n_rows)

    for j in range(n_cols):
        if j in bridge_cols:
            # bridge column: coverage spans all clusters uniformly at random,
            # at BRIDGE_DENSITY so it couples without dominating
            target_degree = max(1, int(BRIDGE_DENSITY * n_rows))
            degree = min(rng.poisson(target_degree) + 1, n_rows)
            chosen = rng.choice(all_rows, size=min(degree, n_rows), replace=False)
        else:
            # home column: coverage confined to its own cluster's rows, at
            # LOCAL_DENSITY relative to that cluster's size (not a fixed
            # absolute degree, so it doesn't blow up as cluster size changes)
            home_rows = rows_by_cluster[col_cluster[j]]
            if len(home_rows) == 0:
                continue
            target_degree = max(1, int(LOCAL_DENSITY * len(home_rows)))
            degree = min(rng.poisson(target_degree) + 1, len(home_rows))
            chosen = rng.choice(home_rows, size=degree, replace=False)
        for i in chosen:
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
        "n_rows": n_rows, "n_cols": n_cols, "n_clusters": n_clusters,
        "n_bridge_cols": len(bridge_cols), "n_home_cols": n_cols - len(bridge_cols),
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
