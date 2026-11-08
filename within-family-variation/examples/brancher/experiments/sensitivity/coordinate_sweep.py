"""
Coordinate sweep: vary one parameter of LLM4Branch's released policy at a
time (holding the other 6 fixed at their original values), evaluate on the
held-out easy test set, and see how much performance moves -- Phase 1's
per-dimension sensitivity characterization.

Usage:
    python coordinate_sweep.py                      # full run: 7 params x 6-point grid x 80 instances
    python coordinate_sweep.py --instance_num 5      # quick smoke test
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program
from tester_parallel import evaluate

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "setcover", "program.py")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp_programs")
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def write_program(theta, used_features, bound_params, out_path):
    with open(PROGRAM_PATH) as f:
        original_source = f.read()
    body = original_source.split("\n", 3)[3]  # drop the USED_FEATURES/PARAMS/BOUNDS header lines
    with open(out_path, "w") as f:
        f.write(f"USED_FEATURES = {used_features}\n")
        f.write(f"PARAMS = {list(theta)}\n")
        f.write(f"BOUNDS = {bound_params}\n")
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--cores_list", default="0-5")
    parser.add_argument("--instance_num", type=int, default=80)
    parser.add_argument("--level", default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    start, end = args.cores_list.split("-")
    cores_list = list(range(int(start), int(end) + 1))

    _, used_features, original_theta, bound_params = load_program(PROGRAM_PATH)
    print(f"Original theta: {original_theta}")

    results_path = os.path.join(os.path.dirname(__file__), f"coordinate_sweep_results_{args.level}.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_index", "swept_value", "mean_time", "mean_nodes"])

        for param_index in range(len(original_theta)):
            for value in GRID:
                theta_prime = list(original_theta)
                theta_prime[param_index] = value

                tmp_path = os.path.join(TMP_DIR, f"program_p{param_index}_v{value}.py")
                write_program(theta_prime, used_features, bound_params, tmp_path)

                lock_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "cpu_locks", f"sweep_{param_index}_{value}")
                os.makedirs(lock_dir, exist_ok=True)

                print(f"param[{param_index}]={value}: running...", flush=True)
                result = evaluate(
                    program_path=tmp_path,
                    dataset="setcover",
                    level=args.level,
                    cores_list=cores_list,
                    lock_dir=lock_dir,
                    methods=["evolve"],
                    max_workers=args.max_workers,
                    instance_num=args.instance_num,
                    seed=999999999,
                    save_csv=False,
                    log_level="WARNING",
                    log_file=None,
                )
                mean_time = result["mean_time"][0]
                mean_nodes = result["mean_nodes"][0]
                print(f"  -> mean_time={mean_time:.3f}, mean_nodes={mean_nodes:.1f}", flush=True)
                writer.writerow([param_index, value, mean_time, mean_nodes])
                f.flush()

    print(f"Saved: {results_path}")
