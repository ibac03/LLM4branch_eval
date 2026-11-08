"""
BO-tune SCIP's own relpscost (reliability pseudocost branching, "RPB" in
this project) branching-rule parameters per structural regime (BA/RR/WS),
mirroring the exact BO recipe used throughout this project for the
evolved-policy retunes (tune_evolved_rr_on_ba.py / tune_evolved_ws_on_ba.py
etc.): skopt Optimizer, gp_hedge, 50 calls, natural D_sub=[0,1,2,3] training
seeds, minimizing mean node count on each domain's own cheap training tier.

Unlike the evolved policies (which tune weights in a hand/LLM-designed
score_function), RPB's "parameters" here are real SCIP branching-rule
knobs that combine into its own internal score: pscostweight,
conflictweight, inferenceweight, cutoffweight (the four main scoring-
weight terms) plus minreliable (how many strong-branching calls a
variable needs before its pseudocost is trusted over a fresh strong-
branching evaluation -- a real, well-known-important lever for
relpscost's speed/quality tradeoff).

Training tiers (same easier-than-test tiers used throughout this project):
  BA: n_nodes=400, affinity=5   (category_k_indset_affinity.generate_instance)
  RR: n_nodes=200, degree=3     (category_rr_indset.generate_instance)
  WS: n_nodes=200, k=20, p=0.2  (category_ws_indset.generate_instance)

Usage:
    python tune_rpb.py --domain ba --n_calls 50 --n_initial 10
    python tune_rpb.py --domain rr --n_calls 50 --n_initial 10
    python tune_rpb.py --domain ws --n_calls 50 --n_initial 10
"""
import argparse
import json
import multiprocessing
import os
import sys

import ecole
from skopt import Optimizer
from skopt.space import Real

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import numerical_mean_stable, get_scip_params
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

TRAIN_SEEDS = [0, 1, 2, 3]  # same natural D_sub=4 convention as every other tuning run this session
MIXED_SEEDS_PER_DOMAIN = [0, 1, 2]  # "mixed" domain: 3 seeds x 3 domains = 9 total training instances
FAILURE_PENALTY = 1e9

# (name, lower, upper, default) -- defaults are SCIP's own, used as one of
# the BO's initial candidate points isn't required by skopt but documented
# here for reference.
PARAM_SPACE = [
    ("pscostweight", 0.0, 2.0),      # default 1.0
    ("conflictweight", 0.0, 1.0),    # default 0.01
    ("inferenceweight", 0.0, 1.0),   # default 0.0001
    ("cutoffweight", 0.0, 1.0),      # default 0.0001
    ("minreliable", 0.0, 8.0),       # default 1.0 -- strong-branching reliability threshold
]

DOMAIN_TRAIN_TIME_LIMIT = 180  # same cap used for every other domain-training-tier tuning run


def make_train_instance(domain, seed):
    if domain == "ba":
        return generate_ba_instance(seed=seed, n_nodes=400, affinity=5)
    elif domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=200, degree=3)
    elif domain == "ws":
        return generate_ws_instance(seed=seed, n_nodes=200, k=20, p=0.2)
    raise ValueError(f"unknown domain {domain!r}")


def build_tasks(domain, param_values, train_seeds=None, mixed_seeds_per_domain=None):
    """Normal domains ('ba'/'rr'/'ws') use train_seeds (default TRAIN_SEEDS)
    on that one domain. 'mixed' pools mixed_seeds_per_domain (default
    MIXED_SEEDS_PER_DOMAIN) seeds from ALL THREE domains into one combined
    training set, so the resulting tuning is not specialized to any single
    structural regime -- this is the mixed-model-training-baseline analog
    for RPB (see portfolio.md). Overridable via CLI (--train_seeds_per_domain)
    for the "more tuning samples/budget" scale-up experiment, without
    changing the original small-scale runs' documented defaults."""
    train_seeds = train_seeds if train_seeds is not None else TRAIN_SEEDS
    mixed_seeds_per_domain = mixed_seeds_per_domain if mixed_seeds_per_domain is not None else MIXED_SEEDS_PER_DOMAIN
    if domain == "mixed":
        return [(d, seed, param_values) for d in ("ba", "rr", "ws") for seed in mixed_seeds_per_domain]
    return [(domain, seed, param_values) for seed in train_seeds]


def solve_one(args):
    domain, seed, param_values = args
    instance = make_train_instance(domain, seed)
    scip_params = {
        **get_scip_params(time_limit=DOMAIN_TRAIN_TIME_LIMIT),
        "branching/relpscost/priority": 9999999,
    }
    for (name, _lo, _hi), value in zip(PARAM_SPACE, param_values):
        scip_params[f"branching/relpscost/{name}"] = float(value)

    env = ecole.environment.Configuring(scip_params=scip_params)
    env.seed(0)
    try:
        env.reset(instance)
        _, _, _, _, _ = env.step({})
        m = env.model.as_pyscipopt()
        return {"validity": True, "seed": seed, "nodes": m.getNNodes()}
    except Exception as e:
        return {"validity": False, "seed": seed, "error_info": str(e)}


def objective(param_values, domain, pool, train_seeds=None, mixed_seeds_per_domain=None):
    tasks = build_tasks(domain, param_values, train_seeds, mixed_seeds_per_domain)
    results = pool.map(solve_one, tasks)
    for r in results:
        if not r["validity"]:
            print(f"  FAILED seed={r.get('seed')}: {r.get('error_info')}", flush=True)
            return FAILURE_PENALTY
    mean_nodes = numerical_mean_stable([r["nodes"] for r in results])
    named = {name: round(float(v), 4) for (name, _, _), v in zip(PARAM_SPACE, param_values)}
    print(f"  params={named} -> mean_nodes={mean_nodes:.2f} (n_instances={len(tasks)})", flush=True)
    return mean_nodes


def tune(domain, max_workers, n_calls, n_initial, random_state, train_seeds=None, mixed_seeds_per_domain=None):
    param_space = [Real(lo, hi) for _name, lo, hi in PARAM_SPACE]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=random_state)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        x0 = opt.ask(n_initial)
        y0 = [objective(x, domain, pool, train_seeds, mixed_seeds_per_domain) for x in x0]
        opt.tell(x0, y0)
        for i in range(n_initial, n_calls):
            print(f"RPB-on-{domain} tuning: iteration {i + 1}/{n_calls}", flush=True)
            next_x = opt.ask()
            f_val = objective(next_x, domain, pool, train_seeds, mixed_seeds_per_domain)
            opt.tell(next_x, f_val)

    result = opt.get_result()
    return result.x, result.fun


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ba", "rr", "ws", "mixed"], required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--n_calls", type=int, default=50)
    parser.add_argument("--n_initial", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--out_suffix", default="",
                         help="appended to the output filename, e.g. '_run2', so repeated "
                              "tuning runs (different --random_state) don't overwrite each other.")
    parser.add_argument("--train_seeds_per_domain", type=int, default=None,
                         help="overrides TRAIN_SEEDS to range(train_seeds_per_domain) for ba/rr/ws, "
                              "and MIXED_SEEDS_PER_DOMAIN to the same range for 'mixed' -- for the "
                              "'more tuning samples/budget' scale-up experiment (default: use the "
                              "original small-scale TRAIN_SEEDS=[0,1,2,3] / MIXED_SEEDS_PER_DOMAIN=[0,1,2]).")
    args = parser.parse_args()

    train_seeds = list(range(args.train_seeds_per_domain)) if args.train_seeds_per_domain else None
    mixed_seeds = train_seeds  # same override range, applied per-domain within "mixed" too

    train_desc = (f"train_seeds_per_domain={args.train_seeds_per_domain} (OVERRIDE)" if args.train_seeds_per_domain
                  else (f"MIXED_SEEDS_PER_DOMAIN={MIXED_SEEDS_PER_DOMAIN} x 3 domains" if args.domain == "mixed"
                        else f"natural D_sub={TRAIN_SEEDS}"))
    print(f"Tuning RPB (relpscost) on {args.domain} ({train_desc}), "
          f"n_calls={args.n_calls}, random_state={args.random_state}", flush=True)
    theta, fun = tune(args.domain, args.max_workers, args.n_calls, args.n_initial, args.random_state,
                       train_seeds, mixed_seeds)
    named = {name: float(v) for (name, _, _), v in zip(PARAM_SPACE, theta)}
    print(f"\nDONE: best mean_nodes={fun:.2f}, params={named}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), f"rpb_tuned_{args.domain}{args.out_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(named, f, indent=2)
    print(f"Saved: {out_path}", flush=True)
