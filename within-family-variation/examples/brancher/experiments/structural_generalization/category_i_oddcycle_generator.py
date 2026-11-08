"""
Category I: vertex cover on a disjoint union of small odd cycles, in set-cover
format (rows = edges, columns = vertices; each vertex/column covers its 2
incident edges/rows). This is the classic LP-relaxation-is-symmetric
construction: for an odd cycle, x=0.5 for EVERY vertex is the unique
symmetric LP-optimal solution, so every candidate variable is exactly
0.5-fractional (maximal, tied "centrality") throughout large parts of the
search -- a genuinely different mechanism from every other category tried
(topological symmetry, not density/clustering/multicover).

The hypothesis: LLM4Branch's params[0]/params[1] (fractional-value
"centrality") are likely the single most load-bearing, universally useful
signal for standard SetCover ("branch on the most fractional variable").
If this category makes that signal completely non-discriminating (every
candidate tied at 0.5), a policy re-tuned here has zero incentive to keep
that weight high -- and if it drifts down during BO tuning, that same
policy should measurably regress on standard SetCover, not just be
"less good than sibling retunes" (which is all we've seen so far).

Same n_rows/n_cols scale as the standard "medium" setcover tier (rows and
columns are equal in count here since it's vertex-cover-shaped: n vertices
= n edges in a cycle).
"""
import numpy as np
import pyscipopt

N_ROWS = 1000            # = total edges across all cycles
N_COLS = 1000             # = total vertices across all cycles (1:1 with rows)
CYCLE_LENGTH = 5          # odd, small cycles -- more components = more
                          # symmetric sub-structure repeated throughout the tree
COST_LOW, COST_HIGH = 1.0, 10.0


def generate_instance(seed, n_rows=N_ROWS, n_cols=N_COLS, cycle_length=CYCLE_LENGTH):
    rng = np.random.RandomState(seed)
    assert n_rows == n_cols, "vertex-cover-shaped: rows (edges) == cols (vertices)"
    assert cycle_length % 2 == 1, "cycles must be odd for the x=0.5 LP symmetry"

    n_cycles = n_rows // cycle_length
    n_used = n_cycles * cycle_length  # may be < n_rows if not evenly divisible

    coverage = [set() for _ in range(n_rows)]  # row(edge) -> covering columns(vertices)
    costs = np.zeros(n_cols)
    for c in range(n_cycles):
        base_v = c * cycle_length   # vertex ids for this cycle
        base_e = c * cycle_length   # edge ids for this cycle
        # cost is uniform WITHIN a cycle (varies between cycles) -- this is
        # what preserves the symmetric x=0.5-for-everyone LP relaxation;
        # per-vertex-random costs would break the intra-cycle symmetry and
        # let the LP favor specific vertices instead of tying them all.
        cycle_cost = float(rng.uniform(COST_LOW, COST_HIGH))
        for i in range(cycle_length):
            v1 = base_v + i
            v2 = base_v + (i + 1) % cycle_length
            edge = base_e + i
            coverage[edge].add(v1)
            coverage[edge].add(v2)
            costs[v1] = cycle_cost
            costs[v2] = cycle_cost

    # leftover rows (if n_rows not divisible by cycle_length) get a trivial
    # self-loop-style single-column cover to keep the instance well-formed
    for i in range(n_used, n_rows):
        coverage[i].add(i)
        costs[i] = float(rng.uniform(COST_LOW, COST_HIGH))

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols)]
    for i in range(n_rows):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()

    stats = {
        "n_rows": n_rows, "n_cols": n_cols,
        "n_cycles": n_cycles, "cycle_length": cycle_length,
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_rows)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")
