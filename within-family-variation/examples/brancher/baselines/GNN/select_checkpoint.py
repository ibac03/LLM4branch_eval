"""
Select the GNN baseline's final checkpoint by actual solving time on held-out
BA instances, not by validation loss/accuracy -- same protocol as
MLP/select_checkpoint.py (per the paper's Appendix A finding that
loss/accuracy-optimal checkpoints don't predict the fastest solving
checkpoint).

Uses raw (unnormalized) ecole.observation.NodeBipartite features, matching
exactly how generate_data_ba.py produced the training data -- unlike the
MLP baseline (which uses the Flash/Khalil feature set with normalize()),
the GNN baseline never normalizes features, following Gasse et al. 2019.

Held-out instances: same n_nodes=400,affinity=5 tier training used, but a
disjoint seed range (default starting at 999999999, distinct from training's
0.. and val's 100000000.. ranges).
"""
import argparse
import glob
import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, geometric_mean_stable

from model import GNNPolicy

BA_N_NODES = 400
BA_AFFINITY = 5
BA_TIME_LIMIT = 180

ecole = None


def _init_worker(import_lock):
    global ecole
    with import_lock:
        import ecole as _ecole
    ecole = _ecole


def make_ba_instance(seed):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_N_NODES, affinity=BA_AFFINITY,
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def evaluate_one(args):
    checkpoint_path, seed = args
    model = GNNPolicy()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_obs_fn = ecole.observation.NodeBipartite()
    scip_params = get_scip_params(time_limit=BA_TIME_LIMIT)
    env = ecole.environment.Branching(observation_function=node_obs_fn, scip_params=scip_params)
    env.seed(0)

    instance = make_ba_instance(seed)
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
    status = scip_model.getStatus()
    return {"seed": seed, "proc_time": proc_time, "nodes": scip_model.getNNodes(), "status": status}


def evaluate_checkpoint(checkpoint_path, seeds, max_workers, import_lock):
    tasks = [(checkpoint_path, seed) for seed in seeds]
    with Pool(processes=max_workers, initializer=_init_worker, initargs=(import_lock,)) as pool:
        results = pool.map(evaluate_one, tasks)
    mean_time = geometric_mean_stable([r["proc_time"] for r in results])
    mean_nodes = geometric_mean_stable([r["nodes"] for r in results])
    return mean_time, mean_nodes, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=os.path.join(os.path.dirname(__file__), "checkpoints_ba"))
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--num_instances", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=999999999)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.num_instances))
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "epoch_*.pt")))
    print(f"Evaluating {len(checkpoints)} checkpoints on {len(seeds)} held-out BA instances "
          f"(n_nodes={BA_N_NODES}, affinity={BA_AFFINITY}, time_limit={BA_TIME_LIMIT}s)...", flush=True)

    multiprocessing.set_start_method("spawn", force=True)
    import_lock = multiprocessing.Lock()

    best = None
    for ckpt_path in checkpoints:
        mean_time, mean_nodes, _ = evaluate_checkpoint(ckpt_path, seeds, args.max_workers, import_lock)
        epoch = os.path.basename(ckpt_path)
        print(f"{epoch}: mean_proc_time={mean_time:.4f}s, mean_nodes={mean_nodes:.1f}", flush=True)
        if best is None or mean_time < best[1]:
            best = (ckpt_path, mean_time, mean_nodes)

    print(f"\nTIME-OPTIMAL CHECKPOINT: {best[0]} (mean_proc_time={best[1]:.4f}s, mean_nodes={best[2]:.1f})",
          flush=True)
