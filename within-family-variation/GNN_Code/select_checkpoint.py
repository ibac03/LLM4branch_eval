"""
Select a GNN baseline's final checkpoint by actual solving time on held-out
instances, not by validation loss/accuracy (loss/accuracy-optimal
checkpoints don't reliably predict the fastest solving checkpoint).

Uses raw (unnormalized) ecole.observation.NodeBipartite features, matching
exactly how generate_data.py produced the training data -- unlike a
Flash/Khalil-feature MLP baseline, this GNN never normalizes features,
following Gasse et al. 2019.

Evaluates on each domain's own TRAINING tier (cheap/fast, same tier
generate_data.py trained on) at a seed range (default starting at
500000000) that is deliberately distinct from BOTH the training data's own
seed ranges (0.. for train, 100000000.. for val) AND the true held-out TEST
seeds used in crosstest.py (999999999.. for BA, 1000.. for RR, 200.. for
WS) -- this is purely an internal, cheap checkpoint-selection step, not the
final reported result. Run crosstest.py separately for the real numbers.
"""
import argparse
import glob
import multiprocessing
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from scip_utils import get_scip_params, geometric_mean_stable
from instance_generators import make_train_instance

from model import GNNPolicy

SELECTION_TIME_LIMIT = 180

ecole = None


def _init_worker(import_lock):
    global ecole
    with import_lock:
        import ecole as _ecole
    ecole = _ecole


def evaluate_one(args):
    checkpoint_path, domain, seed = args
    model = GNNPolicy()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_obs_fn = ecole.observation.NodeBipartite()
    scip_params = get_scip_params(time_limit=SELECTION_TIME_LIMIT)
    env = ecole.environment.Branching(observation_function=node_obs_fn, scip_params=scip_params)
    env.seed(0)

    instance = make_train_instance(domain, seed)
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


def evaluate_checkpoint(checkpoint_path, domain, seeds, max_workers, import_lock):
    tasks = [(checkpoint_path, domain, seed) for seed in seeds]
    with Pool(processes=max_workers, initializer=_init_worker, initargs=(import_lock,)) as pool:
        results = pool.map(evaluate_one, tasks)
    mean_time = geometric_mean_stable([r["proc_time"] for r in results])
    mean_nodes = geometric_mean_stable([r["nodes"] for r in results])
    return mean_time, mean_nodes, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ba", "rr", "ws"], required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--num_instances", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=500000000)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(__file__), f"checkpoints_{args.domain}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_instances))
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    print(f"Evaluating {len(checkpoints)} checkpoints ({args.domain}) on {len(seeds)} held-out "
          f"{args.domain}-training-tier instances...", flush=True)

    multiprocessing.set_start_method("spawn", force=True)
    import_lock = multiprocessing.Lock()

    best = None
    for ckpt_path in checkpoints:
        mean_time, mean_nodes, _ = evaluate_checkpoint(ckpt_path, args.domain, seeds, args.max_workers, import_lock)
        epoch = os.path.basename(ckpt_path)
        print(f"{epoch}: mean_proc_time={mean_time:.4f}s, mean_nodes={mean_nodes:.1f}", flush=True)
        if best is None or mean_time < best[1]:
            best = (ckpt_path, mean_time, mean_nodes)

    print(f"\nTIME-OPTIMAL CHECKPOINT ({args.domain}): {best[0]} "
          f"(mean_proc_time={best[1]:.4f}s, mean_nodes={best[2]:.1f})", flush=True)
