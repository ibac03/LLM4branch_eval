"""
Category K: IndependentSet with elevated Barabasi-Albert affinity.

Unlike SetCover (LLM4Branch's own generator uses uniform random density --
no natural degree heterogeneity, which is why Category A/G/J needed
hand-built hub/niche structure), IndependentSet's own generator already
uses a Barabasi-Albert graph, which is NATIVELY heavy-tailed: a few hub
(high-degree) nodes, many niche (low-degree) nodes, controlled by the
`affinity` parameter (avg edges each new node attaches with).

LLM4Branch's released IndSet policy (program/indset/program.py) positively
weights constraint_count (params[3]=0.946 -- same generic "touches many
constraints is a good branching signal" bias seen in SetCover's param[6]).
constraint_count for an IndSet variable IS its node degree. Raising
affinity makes hub nodes far more dominant/high-degree relative to niche
nodes, testing whether a policy retuned for a hub-heavy graph regime
loses its edge on LLM4Branch's own standard affinity=4 setting.

n_nodes stays matched to LLM4Branch's own tiers (DATASET_PARAMS['indset']):
easy=750, medium=1000 -- so results are directly comparable to the
paper's own affinity=4 numbers at the same sizes.
"""
import ecole

N_NODES_MEDIUM = 1000
N_NODES_EASY = 750
AFFINITY_STANDARD = 4   # LLM4Branch's own default (all tiers)
AFFINITY_HIGH = 12      # Category K: candidate elevated-affinity setting


def generate_instance(seed, n_nodes=N_NODES_MEDIUM, affinity=AFFINITY_HIGH):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=n_nodes, affinity=affinity,
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


if __name__ == "__main__":
    inst = generate_instance(seed=0)
    m = inst.as_pyscipopt()
    print(f"n_vars={m.getNVars()}, n_conss={m.getNConss()}")
