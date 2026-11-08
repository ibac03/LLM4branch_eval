"""
Standalone Maximum Independent Set instance generators for all three
structural regimes (BA, RR, WS), for both the cheap TRAINING tier (used by
generate_data.py) and the held-out TEST tier (used by select_checkpoint.py
and crosstest.py). Self-contained -- no dependency on the rest of the
LLM4Branch repo.

--------------------------------------------------------------------------
Why two different tiers per domain (training vs. test):
--------------------------------------------------------------------------
Training on the SAME instance size/difficulty as the held-out test set is
prohibitively slow for data generation (each sample requires running
strong-branching, which is far more expensive per node than any fast
branching rule). The training tier for each domain below is a smaller/
cheaper instance size deliberately calibrated to still produce a
meaningfully deep search tree (not trivially easy), reusing the exact
tiers already established elsewhere in this project's evolved-policy
pipeline:
  - BA:  n_nodes=400, affinity=5   (calibrated tractable BA training tier)
  - RR:  n_nodes=200, degree=3     (RR's own evolution-training tier)
  - WS:  n_nodes=200, k=20, p=0.2  (WS's own evolution-training tier)

The TEST tier for each domain is the harder, calibrated held-out
difficulty used throughout the paper's oracle-gap table:
  - BA:  n_nodes=1000, affinity=4 ("medium" tier, ecole's own IndSet
         generator, matching the paper's own convention), seeds
         999999999..999999999+29 (30 instances)
  - RR:  n_nodes=250, degree=3, seeds 1000..1029 (30 instances)
  - WS:  n_nodes=200, k=20, p=0.25, seeds 200..229 (30 instances)

All three domains are Maximum Independent Set; RR and WS are built
directly via networkx + pyscipopt (ecole has no regular-graph or
small-world generator), matching how the main repo's category_rr_indset.py
/ category_ws_indset.py build them. BA uses ecole's own
IndependentSetGenerator with graph_type=barabasi_albert.
"""
import ecole
import networkx as nx
import pyscipopt

# ---------------------------------------------------------------------
# Training tiers (cheap, used only for imitation-learning data generation)
# ---------------------------------------------------------------------
BA_TRAIN_N_NODES, BA_TRAIN_AFFINITY = 400, 5
RR_TRAIN_N_NODES, RR_TRAIN_DEGREE = 200, 3
WS_TRAIN_N_NODES, WS_TRAIN_K, WS_TRAIN_P = 200, 20, 0.2

# ---------------------------------------------------------------------
# Test tiers (held-out, used for checkpoint selection and the final
# cross-domain evaluation -- these are the EXACT same instances/seeds
# used throughout the rest of the oracle-gap table for the other
# (non-GNN) fixed policies, so results are directly comparable)
# ---------------------------------------------------------------------
BA_TEST_N_NODES, BA_TEST_AFFINITY = 1000, 4
BA_TEST_SEEDS = list(range(999999999, 999999999 + 30))

RR_TEST_N_NODES, RR_TEST_DEGREE = 250, 3
RR_TEST_SEEDS = list(range(1000, 1030))

WS_TEST_N_NODES, WS_TEST_K, WS_TEST_P = 200, 20, 0.25
WS_TEST_SEEDS = list(range(200, 230))


def make_ba_instance(seed, n_nodes, affinity):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=n_nodes, affinity=affinity,
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def make_rr_instance(seed, n_nodes, degree):
    d, n = degree, n_nodes
    if (n * d) % 2 != 0:
        n += 1  # networkx requires n*d even; bump by one node (negligible size effect)
    g = nx.random_regular_graph(d, n, seed=seed)
    return _mis_model_from_graph(g)


def make_ws_instance(seed, n_nodes, k, p):
    g = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)
    return _mis_model_from_graph(g)


def _mis_model_from_graph(g):
    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{v}", vtype="B", obj=1.0) for v in g.nodes()]
    for u, v in g.edges():
        model.addCons(variables[u] + variables[v] <= 1)
    model.setMaximize()
    return ecole.scip.Model.from_pyscipopt(model)


def make_train_instance(domain, seed):
    """domain in {'ba', 'rr', 'ws'} -- cheap training-tier instance."""
    if domain == "ba":
        return make_ba_instance(seed, BA_TRAIN_N_NODES, BA_TRAIN_AFFINITY)
    elif domain == "rr":
        return make_rr_instance(seed, RR_TRAIN_N_NODES, RR_TRAIN_DEGREE)
    elif domain == "ws":
        return make_ws_instance(seed, WS_TRAIN_N_NODES, WS_TRAIN_K, WS_TRAIN_P)
    raise ValueError(f"unknown domain {domain!r}, expected 'ba'/'rr'/'ws'")


def make_test_instance(domain, seed):
    """domain in {'ba', 'rr', 'ws'} -- held-out test-tier instance."""
    if domain == "ba":
        return make_ba_instance(seed, BA_TEST_N_NODES, BA_TEST_AFFINITY)
    elif domain == "rr":
        return make_rr_instance(seed, RR_TEST_N_NODES, RR_TEST_DEGREE)
    elif domain == "ws":
        return make_ws_instance(seed, WS_TEST_N_NODES, WS_TEST_K, WS_TEST_P)
    raise ValueError(f"unknown domain {domain!r}, expected 'ba'/'rr'/'ws'")


TEST_SEEDS = {"ba": BA_TEST_SEEDS, "rr": RR_TEST_SEEDS, "ws": WS_TEST_SEEDS}

# Per-domain SCIP time cap for the FINAL held-out test evaluation, matching
# exactly what was used to produce every other row of the same oracle-gap
# table (Original / Evolved-RR / Evolved-WS) in the main repo: BA follows
# the paper's own 1-hour-per-instance convention (tester_parallel.py's
# default get_scip_params() with no time_limit arg -> 3600s); RR and WS
# both use the 120s cap established in experiments/structural_generalization/
# eval_category_b.py (used throughout that pipeline's RR/WS crosstests).
# Keep these consistent with the rest of the table -- do not change without
# re-running the other rows too.
TEST_TIME_LIMIT = {"ba": 3600, "rr": 120, "ws": 120}


if __name__ == "__main__":
    # Quick self-check: print instance size for each domain's train/test tier.
    for domain in ["ba", "rr", "ws"]:
        train_inst = make_train_instance(domain, seed=0).as_pyscipopt()
        test_inst = make_test_instance(domain, seed=TEST_SEEDS[domain][0]).as_pyscipopt()
        print(f"{domain}: train n_vars={train_inst.getNVars()} n_conss={train_inst.getNConss()} | "
              f"test n_vars={test_inst.getNVars()} n_conss={test_inst.getNConss()} "
              f"(seeds {TEST_SEEDS[domain][0]}..{TEST_SEEDS[domain][-1]}, "
              f"{len(TEST_SEEDS[domain])} instances)")
