"""
Random perturbations at several radii: nudge ALL 7 parameters of LLM4Branch's
released policy simultaneously (each coordinate independently perturbed by
Uniform(-r, +r), reflected back into [0,1] bounds), at a few increasing radii,
to see how quickly performance degrades under combined, realistic imprecision
-- the second half of Phase 1's perturbation-sensitivity characterization
(the coordinate sweep did one dimension at a time; this does all 7 at once).

Multiple random samples are drawn per radius, since a single random direction
could get lucky or unlucky -- we want the actual spread of outcomes at each
noise level, not one arbitrary point. One deliberate "worst-case" point is
also evaluated per radius: params[2] and params[6] (the two dominant
parameters identified by the coordinate sweep) pushed simultaneously toward
their known-worst direction (both sweeps found value=0.0 worst), rather than
relying on random sampling to stumble into that combination by luck.

Perturbations use reflection, not clipping, at the [0,1] bounds: params[0]
sits exactly at its upper bound (1.0) in the released policy, so clipping
would make positive perturbations there a no-op, silently understating that
coordinate's sensitivity. Reflection preserves the perturbation's magnitude
instead of truncating it.

Usage:
    python random_perturbation.py
    python random_perturbation.py --instance_num 5   # smoke test
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
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp_programs_perturb")
RADII = [0.025, 0.05, 0.1, 0.15, 0.2]
N_SAMPLES_PER_RADIUS = 10
RNG_SEED = 42
DOMINANT_PARAMS = [2, 6]  # from the coordinate sweep: highest-impact, both worst at value=0.0


def reflect(values, lo, hi):
    """Reflect values back into [lo, hi] instead of clipping, so a
    perturbation's magnitude isn't silently truncated at a boundary."""
    span = hi - lo
    period = 2 * span
    v = (values - lo) % period
    v = np.where(v > span, period - v, v)
    return v + lo


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
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--cores_list", default="0-5")
    parser.add_argument("--instance_num", type=int, default=80)
    parser.add_argument("--level", default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    start, end = args.cores_list.split("-")
    cores_list = list(range(int(start), int(end) + 1))

    _, used_features, original_theta, bound_params = load_program(PROGRAM_PATH)
    original_theta = np.array(original_theta)
    lo = np.array([b[0] for b in bound_params])
    hi = np.array([b[1] for b in bound_params])
    print(f"Original theta: {list(original_theta)}")

    rng = np.random.RandomState(RNG_SEED)

    def run_one(theta_prime, radius, tag, lock_suffix):
        tmp_path = os.path.join(TMP_DIR, f"program_r{radius}_{lock_suffix}.py")
        write_program(theta_prime, used_features, bound_params, tmp_path)

        lock_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "cpu_locks", f"perturb_{radius}_{lock_suffix}")
        os.makedirs(lock_dir, exist_ok=True)

        print(f"radius={radius} {tag}: running...", flush=True)
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
            raise RuntimeError(f"evaluate() failed at radius={radius} {tag}: {result.get('error_info')}")
        mean_time = result["mean_time"][0]
        mean_nodes = result["mean_nodes"][0]
        print(f"  -> mean_time={mean_time:.3f}, mean_nodes={mean_nodes:.1f}", flush=True)
        return mean_time, mean_nodes

    results_path = os.path.join(os.path.dirname(__file__), f"random_perturbation_results_{args.level}.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["radius", "type", "sample_index", "mean_time", "mean_nodes", "theta_prime"])

        for radius in RADII:
            # Deliberate worst-case point: push the two dominant params toward
            # their known-worst direction simultaneously, others left at original.
            theta_worst = original_theta.copy()
            for p in DOMINANT_PARAMS:
                theta_worst[p] = reflect(np.array([original_theta[p] - radius]), lo[p], hi[p])[0]
            mean_time, mean_nodes = run_one(theta_worst, radius, "worst_case", "worst")
            writer.writerow([radius, "worst_case", -1, mean_time, mean_nodes, [float(x) for x in theta_worst]])
            f.flush()

            for sample_index in range(N_SAMPLES_PER_RADIUS):
                noise = rng.uniform(-radius, radius, size=len(original_theta))
                theta_prime = reflect(original_theta + noise, lo, hi)
                mean_time, mean_nodes = run_one(theta_prime, radius, f"sample={sample_index}", f"s{sample_index}")
                writer.writerow([radius, "random", sample_index, mean_time, mean_nodes, [float(x) for x in theta_prime]])
                f.flush()

    print(f"Saved: {results_path}")
