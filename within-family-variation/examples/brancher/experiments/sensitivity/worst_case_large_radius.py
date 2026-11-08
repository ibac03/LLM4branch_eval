"""
Follow-up to random_perturbation.py: the deliberate worst-case point there
only reached ~19.5% of the LLM4Branch-vs-MLP margin at radius=0.2, far short
of the coordinate sweep's 92.5% near-miss -- because radius<=0.2 (a +-20%
nudge) is a much smaller absolute change than the coordinate sweep's full
[0,1]-range single-parameter swing (params[6] alone moved by -0.676).

This tests larger radii for the SAME directed worst-case perturbation
(params[2] and params[6] pushed toward 0, their known-worst value from the
coordinate sweep), using clamping (not reflection) since the goal is to
march toward and saturate at that known extreme, not bounce back past it.
radius=0.7 exceeds both params' original values (0.317, 0.676), so it
reaches the full "both at exactly 0.0 simultaneously" point -- the most
direct combined-perturbation test possible.

Usage:
    python worst_case_large_radius.py --level medium
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program
from tester_parallel import evaluate

PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "program", "setcover", "program.py")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp_programs_worst")
LARGE_RADII = [0.3, 0.5, 0.7]
DOMINANT_PARAMS = [2, 6]


def write_program(theta, used_features, bound_params, out_path):
    with open(PROGRAM_PATH) as f:
        original_source = f.read()
    body = original_source.split("\n", 3)[3]
    with open(out_path, "w") as f:
        f.write(f"USED_FEATURES = {used_features}\n")
        f.write(f"PARAMS = {[float(x) for x in theta]}\n")
        f.write(f"BOUNDS = {bound_params}\n")
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--cores_list", default="0-4")
    parser.add_argument("--instance_num", type=int, default=80)
    parser.add_argument("--level", default="medium", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    start, end = args.cores_list.split("-")
    cores_list = list(range(int(start), int(end) + 1))

    _, used_features, original_theta, bound_params = load_program(PROGRAM_PATH)
    original_theta = np.array(original_theta)
    lo = np.array([b[0] for b in bound_params])
    hi = np.array([b[1] for b in bound_params])
    print(f"Original theta: {list(original_theta)}")

    results_path = os.path.join(os.path.dirname(__file__), f"worst_case_large_radius_results_{args.level}.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["radius", "mean_time", "mean_nodes", "theta_prime"])

        for radius in LARGE_RADII:
            theta_prime = original_theta.copy()
            for p in DOMINANT_PARAMS:
                theta_prime[p] = np.clip(original_theta[p] - radius, lo[p], hi[p])

            tmp_path = os.path.join(TMP_DIR, f"program_r{radius}.py")
            write_program(theta_prime, used_features, bound_params, tmp_path)

            lock_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "cpu_locks", f"worst_large_{radius}")
            os.makedirs(lock_dir, exist_ok=True)

            print(f"radius={radius}: theta={list(theta_prime)}", flush=True)
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
            if not result.get("validity", False):
                raise RuntimeError(f"evaluate() failed at radius={radius}: {result.get('error_info')}")
            mean_time = result["mean_time"][0]
            mean_nodes = result["mean_nodes"][0]
            print(f"  -> mean_time={mean_time:.3f}, mean_nodes={mean_nodes:.1f}", flush=True)
            writer.writerow([radius, mean_time, mean_nodes, [float(x) for x in theta_prime]])
            f.flush()

    print(f"Saved: {results_path}")
