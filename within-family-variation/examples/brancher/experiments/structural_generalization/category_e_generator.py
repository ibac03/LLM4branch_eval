"""
Category E: exact column symmetry.

Standard SetCoverGenerator instances have no symmetric columns -- every
column's coverage set is an independent random draw, so no two columns are
(with overwhelming probability) identical. This generator deliberately
builds groups of columns that are EXACTLY identical (same coverage set,
same cost) -- a classic pathology for generic branch-and-bound: permuting
which member of a symmetric group you pick gives an equivalent solution,
so naive search can explode combinatorially re-exploring symmetric
equivalents unless something (symmetry breaking, or a policy that can
somehow avoid caring) handles it.

The mechanism this targets is sharper than Category A/B/C: LLM4Branch's
score is a deterministic function of a variable's *own* static features.
Members of an exact symmetric group have IDENTICAL features at every node
(same fractional value, same objective coefficient, same pseudocost, same
constraint count) -- so LLM4Branch's formula gives them the exact same
score and has no mechanism whatsoever to break the tie intelligently
(falls back to array order / argmax's first-index tie-break, effectively
arbitrary). RPB's strong-branching-style lookahead does not care about
symmetry either in principle (it would also see identical LP impact from
symmetric candidates) -- so this may not favor RPB over LLM4Branch as
cleanly as hoped; the honest expectation is that symmetry makes the
instance harder for both, and the open empirical question is whether one
degrades more than the other.

Same n_rows/n_cols scale as the standard "medium" setcover tier.
"""
import numpy as np
import pyscipopt

N_ROWS = 500
N_COLS = 500
DENSITY = 0.05             # matches standard's per-column coverage density
GROUP_SIZE = 10            # each symmetric group has this many exactly identical columns
SYMMETRIC_FRACTION = 0.9   # fraction of all columns that belong to a symmetric group
                           # (the rest are ordinary independent random columns)
COST_LOW, COST_HIGH = 1.0, 10.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS):
    rng = np.random.RandomState(seed)

    n_symmetric_cols = int(n_cols * SYMMETRIC_FRACTION)
    n_symmetric_cols -= n_symmetric_cols % GROUP_SIZE  # round down to a multiple of GROUP_SIZE
    n_groups = n_symmetric_cols // GROUP_SIZE
    n_ordinary_cols = n_cols - n_symmetric_cols

    all_rows = np.arange(n_rows)
    coverage = []  # list of sets, one per column
    costs = []

    # symmetric groups: draw ONE coverage set + cost per group, replicate GROUP_SIZE times
    for _ in range(n_groups):
        degree = max(1, int(rng.binomial(n_rows, DENSITY)))
        group_coverage = set(rng.choice(all_rows, size=degree, replace=False).tolist())
        group_cost = float(rng.uniform(COST_LOW, COST_HIGH))
        for _ in range(GROUP_SIZE):
            coverage.append(set(group_coverage))
            costs.append(group_cost)

    # ordinary columns: independent random draws, exactly like standard instances
    for _ in range(n_ordinary_cols):
        degree = max(1, int(rng.binomial(n_rows, DENSITY)))
        coverage.append(set(rng.choice(all_rows, size=degree, replace=False).tolist()))
        costs.append(float(rng.uniform(COST_LOW, COST_HIGH)))

    # shuffle column order so symmetric groups aren't trivially contiguous
    order = rng.permutation(n_cols)
    coverage = [coverage[j] for j in order]
    costs = [costs[j] for j in order]

    # feasibility repair: any row not covered by any column gets a random column extended
    row_covered = [False] * n_rows
    for cov in coverage:
        for i in cov:
            row_covered[i] = True
    for i in range(n_rows):
        if not row_covered[i]:
            j = rng.randint(n_cols)
            coverage[j].add(i)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=costs[j]) for j in range(n_cols)]
    row_to_cols = [[] for _ in range(n_rows)]
    for j, cov in enumerate(coverage):
        for i in cov:
            row_to_cols[i].append(j)
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in row_to_cols[i]) >= 1)
    model.setMinimize()

    stats = {
        "n_rows": n_rows, "n_cols": n_cols,
        "n_symmetric_cols": n_symmetric_cols, "n_groups": n_groups,
        "n_ordinary_cols": n_ordinary_cols,
        "avg_degree": float(np.mean([len(c) for c in coverage])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
