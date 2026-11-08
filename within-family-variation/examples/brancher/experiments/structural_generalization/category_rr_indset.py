"""
Random-regular (RR) graph Maximum Independent Set, per regular.md.

Unlike BA (heavy-tailed/hub-heavy degree distribution) and ER (flat/
Poisson degree distribution, but still with real variance), a random
d-regular graph gives EVERY vertex exactly degree d -- removing degree
heterogeneity entirely at the input level. LLM4Branch's released IndSet
policy weights constraint_count (=degree) at params[3]=0.946, and BA
tuning pushed it further to ~1.956 (per regular.md) -- betting heavily
on degree variation carrying information. RR is the sharpest possible
test of whether that signal is BA-specific: if every vertex has the
same degree, degree itself carries zero information among fresh
variables (though branching/presolve can still create degree drift as
the search progresses -- measured directly, not assumed, in Phase 6).

ecole's IndependentSetGenerator does not support a regular graph_type,
so this builds the graph directly via networkx.random_regular_graph and
constructs the MIS edge-formulation MILP via pyscipopt (max sum x_v,
s.t. x_u + x_v <= 1 for each edge), mirroring how the project's other
custom (non-ecole) generators (Category A/G/J for SetCover) work.
"""
import networkx as nx
import pyscipopt
import ecole


N_NODES_DEFAULT = 250  # calibrated: geomean~29s, all-optimal, up from n=200's ~2-3s (too easy)
DEGREE_DEFAULT = 3


def generate_instance(seed, n_nodes=N_NODES_DEFAULT, degree=DEGREE_DEFAULT):
    """Returns an ecole.scip.Model (converted from pyscipopt), matching
    the return type solve_llm4branch expects."""
    # networkx requires n_nodes*degree to be even
    d = degree
    n = n_nodes
    if (n * d) % 2 != 0:
        n += 1  # bump by one node to make it feasible; negligible size effect
    g = nx.random_regular_graph(d, n, seed=seed)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{v}", vtype="B", obj=1.0) for v in range(n)]
    for u, v in g.edges():
        model.addCons(variables[u] + variables[v] <= 1)
    model.setMaximize()

    return ecole.scip.Model.from_pyscipopt(model)


if __name__ == "__main__":
    inst = generate_instance(seed=0, n_nodes=1000, degree=8)
    m = inst.as_pyscipopt()
    print(f"n_vars={m.getNVars()}, n_conss={m.getNConss()}")
