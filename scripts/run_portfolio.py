"""Solve the same instances with many branching policies -> one CSV row per (policy, env_seed, instance).

    python scripts/run_portfolio.py --level easy --seeds 999999999:200 --policies orig,rr,ws,rpb \
        --env-seeds orig:0,1,2 --cap 300 --cores 8-103 --out runs/task4_easy/portfolio.csv

A policy name is a file in policies/<name>.py (same format as the LLM4Branch programs) or "rpb".
The solve is the same as in the official tester (tester_parallel.solver), with a time limit and a
selectable Ecole seed. Rows already in the CSV are skipped, so the command can run again to continue.
"""
import argparse
import csv
import os
import random
import sys
import time
from functools import partial
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "LLM4Branch", "examples", "brancher"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIELDS = ["policy", "env_seed", "seed", "time", "nodes", "status", "gap", "error"]


def solve(task, level, cap):
    policy, env_seed, seed = task[:3]
    override = task[3] if len(task) > 3 else None  # parameter values to use in place of PARAMS (tuning)
    import ecole
    import numpy as np
    from instances import make_instance
    from utils.utils import get_scip_params, load_program, normalize
    row = dict(policy=policy, env_seed=env_seed, seed=seed, time="", nodes="", status="", gap="", error="")
    try:
        instance = make_instance(level, seed)
        scip_params = get_scip_params(time_limit=cap)
        if policy == "rpb":
            env = ecole.environment.Configuring(scip_params={**scip_params, "branching/relpscost/priority": 9999999})
            env.seed(env_seed)
            t0 = time.process_time()
            env.reset(instance)
            env.step({})
        else:
            score_function, used, params, _ = load_program(os.path.join(ROOT, "policies", policy + ".py"))
            score_function = partial(score_function, params=list(override) if override is not None else params)
            khalil, node = ecole.observation.Khalil2016Flash(), ecole.observation.NodeBipartiteFlash()
            khalil.set_active_features([f - 19 for f in used if f >= 19])
            node.set_active_features([f for f in used if f < 19])
            env = ecole.environment.Branching(observation_function={"Node": node, "Khalil": khalil}, scip_params=scip_params)
            env.seed(env_seed)
            t0 = time.process_time()
            obs, action_set, _, done, _ = env.reset(instance)
            while not done:
                f = np.concatenate((obs["Node"].variable_features[action_set, :], obs["Khalil"].features[action_set, :]), axis=1)
                f[:, used] = np.nan_to_num(f[:, used], nan=0.0, posinf=0.0, neginf=0.0)
                f = normalize(f, used)
                obs, action_set, _, done, _ = env.step(action_set[score_function(f).argmax()])
        row["time"] = round(time.process_time() - t0, 3)
        m = env.model.as_pyscipopt()
        row["nodes"], row["status"], row["gap"] = m.getNNodes(), m.getStatus(), m.getGap()
    except Exception as exc:  # a bad policy must not stop the run
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="easy", help="easy / medium / hard (BA), rr, ws")
    p.add_argument("--seeds", required=True, help="START:N, or a file with one seed per line")
    p.add_argument("--policies", required=True, help="comma list")
    p.add_argument("--env-seeds", default="", help="extra Ecole seeds, e.g. orig:1,2 (all policies always run seed 0)")
    p.add_argument("--cap", type=float, default=300)
    p.add_argument("--cores", default="8-103")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if os.path.exists(a.seeds):
        seeds = [int(x) for x in open(a.seeds).read().split()]
    else:
        start, n = map(int, a.seeds.split(":"))
        seeds = list(range(start, start + n))
    extra = {}
    for part in filter(None, a.env_seeds.split(";")):
        name, vals = part.split(":")
        extra[name] = [int(v) for v in vals.split(",")]
    tasks = [(pol, es, s) for pol in a.policies.split(",") for es in [0] + extra.get(pol, []) for s in seeds]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(r["policy"], int(r["env_seed"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
    tasks = [t for t in tasks if t not in done]
    random.Random(0).shuffle(tasks)
    lo, hi = map(int, a.cores.split("-"))
    os.sched_setaffinity(0, range(lo, hi + 1))
    print(f"{len(tasks)} solves to do ({len(done)} already done), {hi - lo + 1} cores", flush=True)

    new = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as fh, Pool(hi - lo + 1, maxtasksperchild=50) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(partial(solve, level=a.level, cap=a.cap), tasks), 1):
            w.writerow(row)
            fh.flush()
            if i % 500 == 0:
                print(f"{i}/{len(tasks)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"PORTFOLIO_DONE {len(tasks)} solves in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
