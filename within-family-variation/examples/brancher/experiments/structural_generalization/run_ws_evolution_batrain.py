"""
Fresh, independent WS evolution replicate at the BA-training-difficulty-
matched tier (n=300,k=20,p=0.2, via evaluator_ws_batrain.py -- see that
file's docstring for the calibration rationale). Genuinely fresh from
iteration 1 (NOT resumed from the original run's checkpoints), with a
different random seed, to a separate output directory -- this is an
independent replicate, not a continuation of the n=200-trained lineage.
200 iterations directly.
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
INITIAL_PROGRAM = os.path.join(REPO_ROOT, "examples", "brancher", "initial_program.py")
EVALUATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator_ws_batrain.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "config_ws.yaml")
EVALUATOR_CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "evaluator_config.yaml")
CORES_LIST = "0-3"
LOCK_DIR = os.path.join(REPO_ROOT, "examples", "brancher", "tmp", "cpu_locks", "ws_evolution_batrain")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ws_evolution_batrain_output")
MAX_ITERATIONS = 200
RANDOM_SEED = 20260909  # distinct from the original run's default seed (42)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    os.makedirs(LOCK_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    config.max_iterations = MAX_ITERATIONS
    config.random_seed = RANDOM_SEED
    evaluator_config = EvaConfig.from_yaml(EVALUATOR_CONFIG_PATH, "ws", CORES_LIST, LOCK_DIR)
    # Pool-level timeouts MUST stay ABOVE SCIP's own internal limits/time (3600s,
    # set by evaluator's bare get_scip_params() when config.target='nodes').
    # An earlier attempt set stage1_timeout=1800 (BELOW 3600): a solve that ran
    # past 1800s got force-killed by the pool's terminate() while still deep in a
    # native SCIP call, so pool.join() in the evaluator's finally block hung
    # forever -> the whole run froze. Keeping both above 3600 means SCIP always
    # self-terminates cleanly first and the pool never has to force-kill anything.
    evaluator_config._config["stage1_timeout"] = 4500
    evaluator_config._config["stage2_timeout"] = 6000
    config.evaluator.dataset = evaluator_config

    print(f"Initial program: {INITIAL_PROGRAM}", flush=True)
    print(f"Evaluation file: {EVALUATION_FILE}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Max iterations: {config.max_iterations}", flush=True)
    print(f"Random seed: {RANDOM_SEED}", flush=True)

    openevolve = OpenEvolve(
        initial_program_path=INITIAL_PROGRAM,
        evaluation_file=EVALUATION_FILE,
        config=config,
        output_dir=OUTPUT_DIR,
    )

    best_program = await openevolve.run()

    print("\n=== WS_EVOLUTION_BATRAIN_COMPLETE ===", flush=True)
    print(f"Best program metrics:", flush=True)
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}", flush=True)
        else:
            print(f"  {name}: {value}", flush=True)
    print(f"Best program code:\n{best_program.code}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
