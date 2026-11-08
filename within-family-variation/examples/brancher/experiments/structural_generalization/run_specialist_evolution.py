"""
Evolve a specialist for the worst-decile BA instances. Warm-started from
the current best general BA policy (program/indset/program.py) -- refining
an already-good policy for the hard cases, not starting from the weak
generic seed, same logic as the WS warm-start run earlier in this
investigation. Trains against evaluator_specialist.py's 8 hand-curated
n=750 seeds (4 genuine losers + 4 worst-decile-non-losers, calibrated to
reproduce the real n=1000 worst-decile phenomenon).
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from openevolve import OpenEvolve
from openevolve.config import load_config, EvaConfig

REPO_ROOT = "/Users/hs/Desktop/LLM-4-Branch"
INITIAL_PROGRAM = os.path.join(REPO_ROOT, "examples", "brancher", "program", "indset", "program.py")  # warm start
EVALUATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator_specialist.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "config_specialist.yaml")
CORES_LIST_STR = "0-3"
LOCK_DIR = os.path.join(REPO_ROOT, "examples", "brancher", "tmp", "cpu_locks", "specialist_evolution")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specialist_evolution_output")
MAX_ITERATIONS = 200
RANDOM_SEED = 20260915


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    os.makedirs(LOCK_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    config.max_iterations = MAX_ITERATIONS
    config.random_seed = RANDOM_SEED

    start, end = CORES_LIST_STR.split("-")
    evaluator_config = EvaConfig(
        target="nodes",
        train_seed=0, train_num=4,   # unused directly -- evaluator_specialist.py hardcodes its own
        valid_seed=0, valid_num=4,   # curated TRAIN_SEEDS/VALID_SEEDS lists instead of a seed range
        max_workers=8,
        cores_list=list(range(int(start), int(end) + 1)),
        lock_dir=LOCK_DIR,
        dataset="specialist",
        rpb=True,
        parameter_opt=True,
        n_calls=50, n_initial=10,
        # both timeouts kept above SCIP's own internal limits/time (3600s, set
        # by evaluator's bare get_scip_params() for target='nodes') even though
        # our curated seeds solve in well under a minute in the typical case --
        # this is about avoiding the terminate()/join() hang failure mode
        # documented in the RR/WS batrain runs, not about expected duration.
        stage1_timeout=4500, stage2_timeout=6000,
    )
    config.evaluator.dataset = evaluator_config

    print(f"Initial program (WARM START): {INITIAL_PROGRAM}", flush=True)
    print(f"Evaluation file: {EVALUATION_FILE}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Max iterations: {config.max_iterations}", flush=True)
    print(f"Random seed: {RANDOM_SEED}", flush=True)
    print(f"cascade_thresholds: {config.evaluator.cascade_thresholds}", flush=True)

    openevolve = OpenEvolve(
        initial_program_path=INITIAL_PROGRAM,
        evaluation_file=EVALUATION_FILE,
        config=config,
        output_dir=OUTPUT_DIR,
    )

    best_program = await openevolve.run()

    print("\n=== SPECIALIST_EVOLUTION_COMPLETE ===", flush=True)
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}", flush=True)
        else:
            print(f"  {name}: {value}", flush=True)
    print(f"Best program code:\n{best_program.code}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
