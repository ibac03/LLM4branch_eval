"""Run the official LLM4Branch tester (RPB vs the Indset policy) on BA instances.

    python scripts/run_tester.py --level medium --seed 999999999 --n 1300 --log runs/medium_official/tester.log
    python scripts/run_tester.py --level medium --seed 800000000 --n 1300 --log runs/medium_hardik/tester.log

Seeds are seed .. seed+n-1. The official test set starts at 999999999 (tester default);
Hardik's instances are 800000000 .. 800001299. One line per instance goes to the log.
csv=False because csv=True crashes in tester_parallel (its `csv` argument hides the csv module).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "LLM4Branch")
sys.path.insert(0, os.path.join(REPO, "examples", "brancher"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="medium")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=1300)
    p.add_argument("--log", required=True)
    p.add_argument("--cores", default="8-127", help="core range; one solve per core")
    a = p.parse_args()

    log = os.path.abspath(a.log)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    locks = os.path.join(ROOT, "tmp", "locks", os.path.basename(os.path.dirname(log)))
    os.makedirs(locks, exist_ok=True)
    lo, hi = map(int, a.cores.split("-"))
    cores = list(range(lo, hi + 1))

    os.chdir(REPO)
    from tester_parallel import evaluate  # noqa: E402

    t = time.time()
    res = evaluate("./examples/brancher/program/indset/program.py", dataset="indset", level=a.level,
                   cores_list=cores, lock_dir=locks, methods=["relpscost", "evolve"], max_workers=len(cores),
                   instance_num=a.n, seed=a.seed, csv=False, log_level="INFO", log_file=log)
    print(f"done in {(time.time() - t) / 60:.1f} min:", json.dumps(res, default=str), flush=True)


if __name__ == "__main__":  # the tester starts worker processes that import this file
    main()
