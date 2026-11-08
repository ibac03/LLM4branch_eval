"""
Category C: cost-coverage correlation.

Standard SetCoverGenerator instances draw column costs independently of
column coverage (degree) -- cost and coverage-width are uncorrelated random
variables. This generator introduces a deliberate correlation: columns that
cover MORE rows are made systematically CHEAPER (negative cost-degree
correlation), so "greedy" cost-per-row-covered reasoning is misleading in a
specific, structured way -- wide, cheap columns look attractive by almost
any local feature (low obj_coef, touches many constraints), but a policy
that always prefers them may miss that the true optimal cover requires a
more careful combination once the easy wide columns' rows are exhausted
and only expensive narrow columns remain for the residual rows.

This targets a different mechanism than Category A (degree heterogeneity
alone, cost independent of degree) or Category B (clustering/community
structure) -- here EVERY column follows the same degree distribution as
standard instances; only the cost<->degree relationship is structured.

Same n_rows/n_cols scale as the standard "medium" setcover tier.
"""
import numpy as np
import pyscipopt

N_ROWS = 1000
N_COLS = 1000
DENSITY = 0.05           # matches standard SetCoverGenerator's default density
COST_LOW, COST_HIGH = 1.0, 10.0
CORRELATION_STRENGTH = 0.4  # how strongly cost is driven by (inverse) degree RANK vs.
                             # independent noise -- 0 = uncorrelated (standard), 1 = fully rank-determined
                             # (0.8 still caused slow/degenerate solves even after the
                             # rank-based tie fix -- a smooth monotonic cost-degree
                             # relationship can still create near-degenerate LP pivots)


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, density=DENSITY):
    rng = np.random.RandomState(seed)

    # degree distribution identical in shape to standard instances: each column's
    # degree ~ Binomial(n_rows, density) (same as i.i.d. Bernoulli(density) coverage)
    degrees = rng.binomial(n_rows, density, size=n_cols)
    degrees = np.clip(degrees, 1, n_rows)

    coverage = [set() for _ in range(n_rows)]
    all_rows = np.arange(n_rows)
    for j in range(n_cols):
        chosen = rng.choice(all_rows, size=degrees[j], replace=False)
        for i in chosen:
            coverage[i].add(j)

    # feasibility repair: any row still uncovered gets a random column
    for i in range(n_rows):
        if not coverage[i]:
            j = rng.randint(n_cols)
            coverage[i].add(j)
            degrees[j] += 1

    # cost negatively correlated with degree via RANK, not raw degree value.
    # Binomial(n_rows, density) concentrates into a narrow band (e.g. 29-75 out
    # of 1000 columns), so many columns share the exact same degree -- using the
    # raw value directly caused near-tied objective coefficients across large
    # groups of columns, crippling the simplex with LP degeneracy even after
    # blending in noise. Rank has no ties by construction, spreading the
    # correlated component smoothly across all n_cols distinct values.
    ranks = np.argsort(np.argsort(-degrees))  # rank 0 = highest degree
    rank_norm = ranks / max(1, n_cols - 1)     # high degree -> low rank_norm -> low cost
    noise = rng.uniform(0.0, 1.0, size=n_cols)
    blended = CORRELATION_STRENGTH * rank_norm + (1 - CORRELATION_STRENGTH) * noise
    costs = COST_LOW + blended * (COST_HIGH - COST_LOW)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()

    correlation = float(np.corrcoef(degrees, costs)[0, 1])
    stats = {
        "n_rows": n_rows, "n_cols": n_cols,
        "degree_min": int(degrees.min()), "degree_max": int(degrees.max()),
        "degree_mean": float(degrees.mean()),
        "cost_degree_correlation": correlation,
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
