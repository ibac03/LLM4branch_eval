"""
Category K large/matched-size ER test tier.

The original Category K crosstest evaluated ER at n=400 (the same size
used for BO training), while the "standard" BA comparison used
LLM4Branch's own medium tier at n=1000 -- a 2.5x size mismatch that
confounded the comparison (ER "solving fast" partly reflected being a
much smaller problem, not just a different topology).

This tier fixes that: n=1000, matching BA-standard's size exactly.
Density was calibrated (calibrate_k6/k7) to find the largest
edge_probability that still solves to genuine optimality across seeds
without hitting SCIP's time limit:
  p=0.003   -> trivial (root-solves, ~1 node)
  p=0.0036  -> all optimal, mean_nodes~2670, geomean~5.8s  <- CHOSEN
  p=0.0038  -> 2/4 seeds time out
  p=0.004   -> 3/4 seeds time out
"""
import ecole

N_NODES_LARGE = 1000
P_LARGE = 0.0036  # calibrated: largest density that reliably solves to optimality at n=1000

ER = ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi


def generate_instance(seed, n_nodes=N_NODES_LARGE, edge_probability=P_LARGE):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=n_nodes, graph_type=ER, edge_probability=edge_probability)
    generator.seed(seed)
    return next(generator)


if __name__ == "__main__":
    inst = generate_instance(seed=0)
    m = inst.as_pyscipopt()
    print(f"n_vars={m.getNVars()}, n_conss={m.getNConss()}")
