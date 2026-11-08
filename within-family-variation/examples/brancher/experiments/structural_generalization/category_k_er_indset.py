"""
Category K: Independent Set on Erdos-Renyi graphs (vs LLM4Branch's own
Barabasi-Albert standard).

Calibration (calibrate_k5_ba_vs_er_small.py) established this is a
genuine topology-shape shift, not a density effect: at the SAME average
degree (~8), BA solves n=400 instances in geomean 0.83s (all optimal)
while ER times out completely (all 4 seeds hit the 120s cap, >132s
each). Even ER at HALF that average degree (p=0.010, avg_deg~4) is
already 2.5x slower and ~21x more nodes than BA at double the density
(geomean 2.07s vs 0.83s, mean_nodes 1749.5 vs 82.2) -- confirming the
effect is about degree-distribution SHAPE (BA: power-law/hub-heavy,
exponent ~-3 regardless of affinity; ER: flat/Poisson, no natural hubs)
not just how dense the graph is.

LLM4Branch's released IndSet policy (program/indset/program.py)
positively weights constraint_count (=node degree), betting on hub
nodes existing to prioritize. ER removes the target of that bet
entirely -- there IS no natural hub to find, so the branching signal a
policy tuned on BA relies on doesn't transfer, and vice versa.

Training tier: n=400, p=0.010 (genuinely solvable, calibrated).
Medium (evaluation) tier: n=1000 to match LLM4Branch's own standard
IndSet medium tier size (DATASET_PARAMS['indset']['medium']); the exact
edge_probability for that size still needs its own tractability check
before being used for evaluation.
"""
import ecole

N_NODES_TRAIN = 400
P_TRAIN = 0.010
N_NODES_MEDIUM = 1000

ER = ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi


def generate_instance(seed, n_nodes=N_NODES_TRAIN, edge_probability=P_TRAIN):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=n_nodes, graph_type=ER, edge_probability=edge_probability)
    generator.seed(seed)
    return next(generator)


if __name__ == "__main__":
    inst = generate_instance(seed=0)
    m = inst.as_pyscipopt()
    print(f"n_vars={m.getNVars()}, n_conss={m.getNConss()}")
