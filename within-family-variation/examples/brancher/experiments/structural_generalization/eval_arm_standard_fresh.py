"""
Evaluate ONE arm on a fresh, non-overlapping batch of standard SetCover
medium instances (seeds 1000000079-1000000228, 150 instances -- the
original 80-instance batches used seeds 999999999-1000000078).
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tester_parallel import evaluate, configure_logger_for_process

FRESH_SEED_START = 1000000079
FRESH_NUM = 150

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm_name", required=True)
    parser.add_argument("--method", required=True, choices=["relpscost", "evolve"])
    parser.add_argument("--program_path", default=None)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--cores_list", default="0-2")
    args = parser.parse_args()

    start, end = args.cores_list.split("-")
    cores_list = list(range(int(start), int(end) + 1))
    lock_dir = f"./tmp/cpu_locks/fresh_standard_{args.arm_name}"
    os.makedirs(lock_dir, exist_ok=True)
    os.makedirs("./log/eval", exist_ok=True)
    log_file = f"./log/eval/fresh_standard_{args.arm_name}.log"
    configure_logger_for_process("INFO", log_file)

    result = evaluate(
        args.program_path, dataset="setcover", level="medium",
        cores_list=cores_list, lock_dir=lock_dir, methods=[args.method],
        max_workers=args.max_workers, instance_num=FRESH_NUM, seed=FRESH_SEED_START,
        save_csv=True, log_level="INFO", log_file=log_file,
    )
    print(f"RESULT[{args.arm_name}]: {result}", flush=True)

    out_dir = "./experiments/structural_generalization/standard_comparison_fresh"
    os.makedirs(out_dir, exist_ok=True)
    src = f"./result/setcover/medium/{args.method}.csv"
    dst = os.path.join(out_dir, f"{args.arm_name}_medium_fresh.csv")
    os.replace(src, dst)
    print(f"Saved: {dst}", flush=True)
