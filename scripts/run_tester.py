"""Run the official LLM4Branch tester (RPB vs the Indset policy) on Ecole's BA test instances.

    N=1300 python scripts/run_tester.py easy [medium ...]      # CORES=<n> to limit the cores used

Writes results/runs/<level>.log (one line per instance); turn it into figures with
scripts/analyze_repo_log.py. Uses tester_parallel.evaluate with csv=False because csv=True crashes
(the `csv` parameter shadows the csv module).
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "external", "LLM4Branch")
sys.path.insert(0, os.path.join(REPO, "examples/brancher"))
os.chdir(REPO)  # tester_parallel resolves dataset paths relative to the repo
from tester_parallel import evaluate  # noqa: E402

CORES = int(os.environ.get("CORES", os.cpu_count()))
OUT = os.environ.get("OUT", os.path.join(ROOT, "results", "runs"))
LOCKS = os.path.join(ROOT, "tmp", "locks")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOCKS, exist_ok=True)
    for level in sys.argv[1:]:
        t = time.time()
        res = evaluate("./examples/brancher/program/indset/program.py", dataset="indset", level=level,
                       cores_list=list(range(CORES)), lock_dir=LOCKS,
                       methods=["relpscost", "evolve"], max_workers=CORES, instance_num=int(os.environ.get("N", 1300)),
                       csv=False, log_level="INFO", log_file=os.path.join(OUT, f"{level}.log"))
        print(level, "done in", round((time.time() - t) / 60, 1), "min:", json.dumps(res, default=str), flush=True)
