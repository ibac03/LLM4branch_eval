"""Tune the parameters of one policy on many training instances (Bayesian optimization, as in LLM4Branch).

    python scripts/tune_params.py --policy orig --out-policy orig_tuned48 --n-train 48 --calls 24 --cores 8-55

LLM4Branch tunes on 4 instances. One run on one instance changes by a factor of about 1.4 with the
random seed, so 4 instances give a noisy target. This script uses many instances (seeds 700000000+,
not in the test set). Target: mean node count (a solve stops at --cap seconds).
"""
import argparse
import json
import os
import re
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
from skopt import Optimizer
from skopt.space import Real

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_portfolio import solve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="orig")
    p.add_argument("--out-policy", required=True)
    p.add_argument("--level", default="easy")
    p.add_argument("--n-train", type=int, default=48)
    p.add_argument("--train-seed", type=int, default=700000000)
    p.add_argument("--calls", type=int, default=24)
    p.add_argument("--cap", type=float, default=60)
    p.add_argument("--upper", type=float, default=3.0, help="upper bound of each parameter")
    p.add_argument("--start", default="", help="extra start points, e.g. 0,0,1.5,2;1,1,1,1")
    p.add_argument("--cores", default="8-55")
    a = p.parse_args()
    lo, hi = map(int, a.cores.split("-"))
    os.sched_setaffinity(0, range(lo, hi + 1))
    src = open(os.path.join(ROOT, "policies", a.policy + ".py")).read()
    x0 = [float(v) for v in re.search(r"PARAMS\s*=\s*\[(.*?)\]", src).group(1).split(",")]
    seeds = list(range(a.train_seed, a.train_seed + a.n_train))
    opt = Optimizer([Real(0.0, a.upper)] * len(x0), acq_func="gp_hedge", random_state=123)
    log = []

    with Pool(hi - lo + 1) as pool:
        def objective(x):
            rows = pool.map(partial(solve, level=a.level, cap=a.cap), [(a.policy, 0, s, list(x)) for s in seeds])
            nodes = float(np.mean([r["nodes"] for r in rows]))
            log.append({"params": list(x), "mean_nodes": nodes, "mean_time": float(np.mean([float(r["time"]) for r in rows])),
                        "not_solved": sum(r["status"] != "optimal" for r in rows)})
            last_time = log[-1]["mean_time"]
            print(f"{len(log):3d}  nodes {nodes:9.1f}  time {last_time:6.2f}  " + str(np.round(x, 3).tolist()), flush=True)
            return nodes

        starts = [x0] + [[float(v) for v in s.split(",")] for s in a.start.split(";") if s]
        for x in starts:
            opt.tell(x, objective(x))
        for x in opt.ask(6):
            opt.tell(x, objective(x))
        while len(log) < a.calls:
            x = opt.ask()
            opt.tell(x, objective(x))

    best = min(log, key=lambda r: r["mean_nodes"])
    new = re.sub(r"PARAMS\s*=\s*\[.*?\]", "PARAMS = " + json.dumps([round(v, 4) for v in best["params"]]), src, count=1)
    open(os.path.join(ROOT, "policies", a.out_policy + ".py"), "w").write(new)
    json.dump({"start": log[0], "best": best, "all": log, "train_seeds": [seeds[0], seeds[-1]]},
              open(os.path.join(ROOT, "policies", a.out_policy + ".tuning.json"), "w"), indent=1)
    print("TUNE_DONE start:", log[0], "best:", best, flush=True)


if __name__ == "__main__":
    main()
