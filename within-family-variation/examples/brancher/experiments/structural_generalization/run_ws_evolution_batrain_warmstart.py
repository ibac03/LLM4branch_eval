"""
WS follow-up: warm-start a fresh evolution run from the current best BA-
difficulty-matched policy (ws_evolution_batrain_output/best/best_program_tuned.py,
stage1_score=0.7326, BO-tuned) instead of the generic initial_program.py seed.
Same evaluator (evaluator_ws_batrain.py, n=300,k=20,p=0.2), same cascade
gate (config_ws.yaml's cascade_thresholds: [0.85, 0.75] -- UNCHANGED), a
fresh database/output dir so this doesn't clobber the completed run. The
zero-shot check showed the OLD easy-trained (n=200) policy scores WORSE
(0.4494) at this difficulty than the already-evolved 0.7326, so this seeds
from the best available point rather than going backward.
200 iterations (same budget as the original run, for comparability).
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
INITIAL_PROGRAM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ws_evolution_batrain_output", "best", "best_program_tuned.py")
EVALUATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator_ws_batrain.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "config_ws.yaml")
EVALUATOR_CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "brancher", "evaluator_config.yaml")
CORES_LIST = "0-3"
LOCK_DIR = os.path.join(REPO_ROOT, "examples", "brancher", "tmp", "cpu_locks", "ws_evolution_batrain_warmstart")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ws_evolution_batrain_warmstart_output")
MAX_ITERATIONS = 200
RANDOM_SEED = 20260911  # distinct from both prior runs' seed (20260909)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    os.makedirs(LOCK_DIR, exist_ok=True)

    config = load_config(CONFIG_PATH)
    config.max_iterations = MAX_ITERATIONS
    config.random_seed = RANDOM_SEED
    evaluator_config = EvaConfig.from_yaml(EVALUATOR_CONFIG_PATH, "ws", CORES_LIST, LOCK_DIR)
    # Same timeout headroom rule as the original batrain run: both MUST stay
    # above SCIP's own internal limits/time (3600s) or pool.terminate() can
    # force-kill a worker mid-solve and hang pool.join() in the evaluator's
    # finally block.
    evaluator_config._config["stage1_timeout"] = 4500
    evaluator_config._config["stage2_timeout"] = 6000
    config.evaluator.dataset = evaluator_config
    # cascade_thresholds deliberately UNCHANGED from config_ws.yaml's [0.85, 0.75] --
    # this run tests whether warm-starting from the best known point (rather
    # than a lowered gate) lets evolution reach stage 2 on its own.

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

    print("\n=== WS_EVOLUTION_BATRAIN_WARMSTART_COMPLETE ===", flush=True)
    print(f"Best program metrics:", flush=True)
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}", flush=True)
        else:
            print(f"  {name}: {value}", flush=True)
    print(f"Best program code:\n{best_program.code}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
