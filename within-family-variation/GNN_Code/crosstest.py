"""
Full 3x3 cross-test: evaluate all three domain-specific GNNs (trained on
BA, RR, WS respectively) on all three domains' held-out TEST sets (30
instances each, exact same seeds/tiers used throughout the rest of the
oracle-gap table -- see instance_generators.py), producing a 3x3 matrix of
(GNN trained on X) x (tested on Y), plus per-GNN pooled-90-instance
geomeans directly comparable to the Original/Evolved-RR/Evolved-WS rows.

Usage (after training all three domains and picking a checkpoint per
domain via select_checkpoint.py):
    python crosstest.py \
        --ba_checkpoint checkpoints_ba/epoch_XXX.pt \
        --rr_checkpoint checkpoints_rr/epoch_XXX.pt \
        --ws_checkpoint checkpoints_ws/epoch_XXX.pt \
        --max_workers 6

Prints the 3x3 timing matrix, each GNN's pooled-90-instance geomean (via
the exact same shifted geomean formula used throughout the rest of the
project, scip_utils.geometric_mean_stable -- pooled over raw per-instance
times, never a geomean-of-geomeans shortcut), and a per-instance-min
oracle row across the three GNNs.
"""
import argparse
import multiprocessing
import time
from multiprocessing import Pool

import numpy as np
import torch

from scip_utils import get_scip_params, geometric_mean_stable
from instance_generators import make_test_instance, TEST_SEEDS, TEST_TIME_LIMIT

from model import GNNPolicy

ecole = None


def _init_worker(import_lock):
    global ecole
    with import_lock:
        import ecole as _ecole
    ecole = _ecole


def evaluate_one(args):
    checkpoint_path, test_domain, seed = args
    model = GNNPolicy()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_obs_fn = ecole.observation.NodeBipartite()
    scip_params = get_scip_params(time_limit=TEST_TIME_LIMIT[test_domain])
    env = ecole.environment.Branching(observation_function=node_obs_fn, scip_params=scip_params)
    env.seed(0)

    instance = make_test_instance(test_domain, seed)
    proc_time_start = time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    with torch.no_grad():
        while not done:
            var_feats = torch.from_numpy(np.nan_to_num(observation.variable_features, nan=0.0,
                                                         posinf=0.0, neginf=0.0).astype(np.float32))
            row_feats = torch.from_numpy(np.nan_to_num(observation.row_features, nan=0.0,
                                                         posinf=0.0, neginf=0.0).astype(np.float32))
            edge_indices = torch.from_numpy(observation.edge_features.indices.astype(np.int64))
            edge_values = torch.from_numpy(observation.edge_features.values.astype(np.float32))
            scores = model(var_feats, row_feats, edge_indices, edge_values).squeeze(-1).numpy()
            action = action_set[scores[action_set].argmax()]
            observation, action_set, _, done, info = env.step(action)
    proc_time = time.process_time() - proc_time_start
    scip_model = env.model.as_pyscipopt()
    return {"seed": seed, "proc_time": proc_time, "nodes": scip_model.getNNodes(),
            "status": scip_model.getStatus()}


def evaluate_cell(checkpoint_path, test_domain, max_workers, import_lock):
    seeds = TEST_SEEDS[test_domain]
    tasks = [(checkpoint_path, test_domain, seed) for seed in seeds]
    with Pool(processes=max_workers, initializer=_init_worker, initargs=(import_lock,)) as pool:
        results = pool.map(evaluate_one, tasks)
    results.sort(key=lambda r: r["seed"])
    times = [r["proc_time"] for r in results]
    n_optimal = sum(1 for r in results if r["status"] == "optimal")
    return times, n_optimal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ba_checkpoint", required=True)
    parser.add_argument("--rr_checkpoint", required=True)
    parser.add_argument("--ws_checkpoint", required=True)
    parser.add_argument("--max_workers", type=int, default=6)
    args = parser.parse_args()

    checkpoints = {"ba": args.ba_checkpoint, "rr": args.rr_checkpoint, "ws": args.ws_checkpoint}
    domains = ["ba", "rr", "ws"]

    multiprocessing.set_start_method("spawn", force=True)
    import_lock = multiprocessing.Lock()

    # cell_times[trained_on][tested_on] -> list of 30 raw per-instance times
    cell_times = {trained: {} for trained in domains}
    print("=== 3x3 cross-test (rows=GNN trained on, cols=tested on) ===", flush=True)
    for trained_on in domains:
        for tested_on in domains:
            times, n_optimal = evaluate_cell(checkpoints[trained_on], tested_on, args.max_workers, import_lock)
            geo = geometric_mean_stable(times)
            cell_times[trained_on][tested_on] = times
            print(f"GNN-{trained_on} on {tested_on}: geomean={geo:.2f}s "
                  f"optimal={n_optimal}/{len(times)}", flush=True)

    print("\n=== Per-GNN pooled 90-instance geomean (BA+RR+WS test sets) ===", flush=True)
    pooled_rows = {}
    for trained_on in domains:
        pooled = cell_times[trained_on]["ba"] + cell_times[trained_on]["rr"] + cell_times[trained_on]["ws"]
        pooled_rows[trained_on] = pooled
        print(f"GNN-{trained_on}: pooled_geomean={geometric_mean_stable(pooled):.2f}s", flush=True)

    oracle_times = [
        min(pooled_rows["ba"][i], pooled_rows["rr"][i], pooled_rows["ws"][i])
        for i in range(90)
    ]
    print(f"Oracle (per-instance min across the 3 GNNs): "
          f"pooled_geomean={geometric_mean_stable(oracle_times):.2f}s", flush=True)

    win_counts = {trained: 0 for trained in domains}
    for i in range(90):
        best = min(domains, key=lambda d: pooled_rows[d][i])
        win_counts[best] += 1
    print(f"Win counts across 90 instances: {win_counts}", flush=True)
