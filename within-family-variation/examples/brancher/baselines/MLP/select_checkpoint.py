"""
Select the MLP baseline's final checkpoint by actual solving time on held-out
instances, not by validation loss/accuracy -- per the paper's Appendix A
finding that loss/accuracy-optimal checkpoints don't predict the fastest
solving checkpoint.

Domain-parameterized (ba/rr/ws), mirroring GNN/select_checkpoint.py's
convention: evaluates at the SAME cheap training tier each domain trained
on (BA n=300,affinity=5; RR n=200,degree=3; WS n=200,k=20,p=0.2), using a
seed range disjoint from both train (0../50000000..) and val
(100000000../150000000..).
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "structural_generalization"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, normalize, geometric_mean_stable
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

from model import MLPPolicy

NUM_NODE_FEATURES = 19
NUM_KHALIL_FEATURES = 72
USED_FEATURES = list(range(NUM_NODE_FEATURES + NUM_KHALIL_FEATURES))
EVAL_TIME_LIMIT = 180  # same cap as training's TRAIN_TIME_LIMIT in generate_data.py

ecole = None


def _init_worker(import_lock):
    global ecole
    with import_lock:
        import ecole as _ecole
    ecole = _ecole


def make_eval_instance(domain, seed):
    if domain == "ba":
        return generate_ba_instance(seed=seed, n_nodes=300, affinity=5)
    elif domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=200, degree=3)
    elif domain == "ws":
        return generate_ws_instance(seed=seed, n_nodes=200, k=20, p=0.2)
    raise ValueError(f"unknown domain {domain!r}")


def evaluate_one(args):
    checkpoint_path, domain, seed = args
    model = MLPPolicy()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs.set_active_features(list(range(NUM_NODE_FEATURES)))
    flash_khalil_obs.set_active_features(list(range(NUM_KHALIL_FEATURES)))
    scip_params = get_scip_params(time_limit=EVAL_TIME_LIMIT)
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=scip_params,
    )
    env.seed(0)

    instance = make_eval_instance(domain, seed)
    proc_time_start = time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    with torch.no_grad():
        while not done:
            variable_features = np.concatenate(
                (observation["Node"].variable_features[action_set, :],
                 observation["Khalil"].features[action_set, :]),
                axis=1,
            )
            variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
            variable_features = normalize(variable_features.copy(), USED_FEATURES)
            scores = model(torch.from_numpy(variable_features.astype(np.float32))).squeeze(-1).numpy()
            action = action_set[scores.argmax()]
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
    parser.add_argument("--seed_start", type=int, default=999999999)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or os.path.join(os.path.dirname(__file__), f"checkpoints_{args.domain}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_instances))
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    print(f"[{args.domain}] Evaluating {len(checkpoints)} checkpoints on {len(seeds)} "
          f"held-out instances (seed_start={args.seed_start})...", flush=True)

    multiprocessing.set_start_method("spawn", force=True)
    import_lock = multiprocessing.Lock()

    best = None
    for ckpt_path in checkpoints:
        mean_time, mean_nodes, _ = evaluate_checkpoint(ckpt_path, args.domain, seeds, args.max_workers, import_lock)
        epoch = os.path.basename(ckpt_path)
        print(f"[{args.domain}] {epoch}: mean_proc_time={mean_time:.4f}s, mean_nodes={mean_nodes:.1f}", flush=True)
        if best is None or mean_time < best[1]:
            best = (ckpt_path, mean_time, mean_nodes)

    print(f"\n[{args.domain}] TIME-OPTIMAL CHECKPOINT: {best[0]} "
          f"(mean_proc_time={best[1]:.4f}s, mean_nodes={best[2]:.1f})", flush=True)
