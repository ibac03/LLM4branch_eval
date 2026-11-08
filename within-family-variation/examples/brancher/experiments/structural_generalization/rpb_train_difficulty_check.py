"""
Sanity check: are BA's n=750,affinity=4 / RR's n=255,degree=3 / WS's
n=240,k=20,p=0.2 training tiers comparably difficult under a policy-
INDEPENDENT baseline (plain RPB, SCIP's default relpscost, no tuning),
not just under each domain's own evolved policy (which could be
circular -- a domain's own policy being fast on its own training tier
doesn't necessarily mean the RAW INSTANCE is comparably hard to BA's).
"""
import sys
import os
import time as _time

import ecole
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, geometric_mean_stable
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

CAP = 180
SEEDS = range(4)


def make_ba_instance(seed):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=750, affinity=4,
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def solve_rpb(instance, time_limit):
    scip_params = {
        **get_scip_params(time_limit=time_limit),
        "branching/relpscost/priority": 9999999,
    }
    env = ecole.environment.Configuring(scip_params=scip_params)
    env.seed(0)
    t0 = _time.process_time()
    env.reset(instance)
    env.step({})
    proc_time = _time.process_time() - t0
    m = env.model.as_pyscipopt()
    return proc_time, m.getNNodes(), m.getStatus()


def probe(label, make_instance):
    times, nodes_list, statuses = [], [], []
    for seed in SEEDS:
        instance = make_instance(seed)
        t, n, s = solve_rpb(instance, CAP)
        times.append(t)
        nodes_list.append(n)
        statuses.append(s)
    print(f"{label}: times={[round(t, 2) for t in times]} statuses={statuses} "
          f"geomean={geometric_mean_stable(times):.2f} mean_nodes={np.mean(nodes_list):.1f}", flush=True)


if __name__ == "__main__":
    print("=== Training-tier difficulty under plain RPB (policy-independent) ===", flush=True)
    probe("BA n=750,affinity=4", make_ba_instance)
    probe("RR n=255,degree=3", lambda seed: generate_rr_instance(seed=seed, n_nodes=255, degree=3))
    probe("WS n=240,k=20,p=0.2", lambda seed: generate_ws_instance(seed=seed, n_nodes=240, k=20, p=0.2))
