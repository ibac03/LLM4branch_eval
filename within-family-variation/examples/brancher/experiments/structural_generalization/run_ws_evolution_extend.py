"""
Hypothesis 2: WS's evolution was cut off at iteration 50 with its best
program found at iteration 39/50 (near the ceiling, unlike RR's early
convergence at iteration 6/50) -- extends it to 200 total iterations by
resuming from checkpoint_50, running 150 more. Same setup as
run_ws_evolution.py otherwise (bypasses openevolve/cli.py's hardcoded
dataset whitelist the same way).
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
EVALUATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator_ws.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "config_ws.yaml")
EVALUATOR_CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "evaluator_config.yaml")
CORES_LIST = "0-3"
LOCK_DIR = os.path.join(REPO_ROOT, "examples", "brancher", "tmp", "cpu_locks", "ws_evolution_extend")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ws_evolution_output")
RESUME_CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoints", "checkpoint_50")
ADDITIONAL_ITERATIONS = 150  # 50 (already done) + 150 = 200 total


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    os.makedirs(LOCK_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    evaluator_config = EvaConfig.from_yaml(EVALUATOR_CONFIG_PATH, "ws", CORES_LIST, LOCK_DIR)
    config.evaluator.dataset = evaluator_config

    print(f"Resuming from: {RESUME_CHECKPOINT}", flush=True)
    print(f"Running {ADDITIONAL_ITERATIONS} additional iterations (target: 200 total)", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)

    openevolve = OpenEvolve(
        initial_program_path=INITIAL_PROGRAM,
        evaluation_file=EVALUATION_FILE,
        config=config,
        output_dir=OUTPUT_DIR,
    )

    best_program = await openevolve.run(iterations=ADDITIONAL_ITERATIONS, checkpoint_path=RESUME_CHECKPOINT)

    print("\n=== WS_EVOLUTION_EXTEND_COMPLETE ===", flush=True)
    print(f"Best program metrics:", flush=True)
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}", flush=True)
        else:
            print(f"  {name}: {value}", flush=True)
    print(f"Best program code:\n{best_program.code}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
