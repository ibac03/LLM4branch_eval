"""
Watts-Strogatz (clustered/small-world) graph Maximum Independent Set --
the third structural regime, per the user's spec.

Unlike RR (uniform degree, locally tree-like, few triangles) and BA
(hub-heavy, unequal degree), a low-rewiring WS graph has uniform-ish
degree (like RR) but STRONG local clustering -- overlapping
neighborhoods, many triangles/short cycles. This isolates whether
specialization is driven by degree distribution alone (RR vs BA) or by
higher-order structure (WS vs RR, same-ish degree, different topology).

networkx.watts_strogatz_graph(n, k, p, seed):
  n: number of nodes
  k: each node initially connected to k nearest neighbors in a ring
     (k must be even); mean degree ~k, same knob as RR's `degree`
  p (beta): rewiring probability. p=0 -> pure ring lattice (maximally
     clustered, regular); p=1 -> approaches a random graph (RR-like).
     Low-to-moderate p keeps most of the ring's clustering while adding
     enough randomness to avoid being trivially symmetric.
"""
import networkx as nx
import pyscipopt
import ecole


N_NODES_DEFAULT = 200  # calibrated: geomean 20-30s at k=20/p=0.25, all-optimal, comparable difficulty for Original and Evolved-RR-tuned-on-BA
K_DEFAULT = 20
P_DEFAULT = 0.25


def generate_instance(seed, n_nodes=N_NODES_DEFAULT, k=K_DEFAULT, p=P_DEFAULT):
    """Returns an ecole.scip.Model (converted from pyscipopt), matching
    the return type solve_llm4branch expects."""
    g = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{v}", vtype="B", obj=1.0) for v in range(n_nodes)]
    for u, v in g.edges():
        model.addCons(variables[u] + variables[v] <= 1)
    model.setMaximize()

    return ecole.scip.Model.from_pyscipopt(model)


if __name__ == "__main__":
    inst = generate_instance(seed=0, n_nodes=200, k=4, p=0.05)
    m = inst.as_pyscipopt()
    print(f"n_vars={m.getNVars()}, n_conss={m.getNConss()}")
