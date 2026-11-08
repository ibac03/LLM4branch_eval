"""
Launcher for RR structural evolution, bypassing openevolve/cli.py's
hardcoded dataset whitelist (['setcover','cauctions','indset','facilities',
'file','user-define']) by building the Config/EvaConfig objects directly
and calling OpenEvolve programmatically, mirroring cli.py's main_async()
logic exactly otherwise.
"""
import asyncio
import logging
import os
import sys

# NOTE: do NOT force a global multiprocessing start method here.
# openevolve's own ProcessPoolExecutor (process_parallel.py) is correctly
# designed for macOS's default 'spawn' -- its _worker_init reconstructs
# everything from serializable dicts, and each worker independently
# re-registers the dynamically-loaded evaluation file under the synthetic
# "evaluation_module" name via Evaluator.__init__. Forcing 'fork' globally
# breaks THAT pool instead (forking from a process with an already-active
# asyncio loop + boto3 threads causes "process pool terminated abruptly"
# crashes). The actual problem is scoped to evaluator_rr.py's own NESTED
# Pool (created inside an already-spawned openevolve worker, to run
# execute_rr across SCIP seeds): ITS grandchild workers, under 'spawn',
# can't resolve "evaluation_module" (only the specific parent worker that
# loaded it did). Fixed locally in evaluator_rr.py via an explicit fork
# context for that inner pool only -- same pattern this session's
# tune_category_*.py scripts used for the same class of problem.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from openevolve import OpenEvolve
from openevolve.config import load_config, EvaConfig

REPO_ROOT = "/Users/hs/Desktop/LLM-4-Branch"
INITIAL_PROGRAM = os.path.join(REPO_ROOT, "examples", "brancher", "initial_program.py")
EVALUATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator_rr.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "config_rr_smoke.yaml")
EVALUATOR_CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "evaluator_config.yaml")
CORES_LIST = "0-3"  # matches max_workers=4 convention used throughout this session
LOCK_DIR = os.path.join(REPO_ROOT, "examples", "brancher", "tmp", "cpu_locks", "rr_evolution")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_evolution_smoke_output")


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    os.makedirs(LOCK_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    evaluator_config = EvaConfig.from_yaml(EVALUATOR_CONFIG_PATH, "rr", CORES_LIST, LOCK_DIR)
    config.evaluator.dataset = evaluator_config

    print(f"Initial program: {INITIAL_PROGRAM}", flush=True)
    print(f"Evaluation file: {EVALUATION_FILE}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Max iterations (config): {config.max_iterations}", flush=True)
    print(f"Dataset config: {evaluator_config._config}", flush=True)

    openevolve = OpenEvolve(
        initial_program_path=INITIAL_PROGRAM,
        evaluation_file=EVALUATION_FILE,
        config=config,
        output_dir=OUTPUT_DIR,
    )

    best_program = await openevolve.run()

    print("\n=== RR_EVOLUTION_COMPLETE ===", flush=True)
    print(f"Best program metrics:", flush=True)
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}", flush=True)
        else:
            print(f"  {name}: {value}", flush=True)
    print(f"Best program code:\n{best_program.code}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
