"""
Generate imitation-learning training data for the GBDT branching baseline
(LightGBM lambdarank over the same 91-dim NodeBipartiteFlash+Khalil2016Flash
feature space the MLP baseline uses -- identical feature extraction, just a
different downstream model). Domain-parameterized (ba/rr/ws), each using its
own cheap training tier -- training at the full test-tier size would hit the
same tractability wall the first GNN data-gen attempt did (strong branching
evaluates every fractional candidate's LP relaxation per node, far more
expensive than any fast branching rule).

Training tiers (same as established throughout this project):
  BA: n_nodes=300, affinity=5   (category_k_indset_affinity.generate_instance --
                                  n=300, not the project's earlier n=400; see
                                  make_train_instance's comment below for why)
  RR: n_nodes=200, degree=3     (category_rr_indset.generate_instance)
  WS: n_nodes=200, k=20, p=0.2  (category_ws_indset.generate_instance)

Usage:
    python generate_data.py --domain ba --split train --num_samples 15000 --seed_start 0         --num_workers 6
    python generate_data.py --domain ba --split val   --num_samples 3000  --seed_start 100000000 --num_workers 6
    (repeat for --domain rr, ws)
"""
import argparse
import gzip
import multiprocessing
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import ecole

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "structural_generalization"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params, normalize
from category_k_indset_affinity import generate_instance as generate_ba_instance
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

NUM_NODE_FEATURES = 19
NUM_KHALIL_FEATURES = 72
USED_FEATURES = list(range(NUM_NODE_FEATURES + NUM_KHALIL_FEATURES))
TRAIN_TIME_LIMIT = 180  # SCIP cap per data-gen episode; see module docstring


def make_train_instance(domain, seed):
    if domain == "ba":
        # n=300 (not the project's usual n=400) -- specifically for data-gen
        # throughput: measured ~5.2x faster strong-branching-based sample
        # generation (95.6 samples/min single-process vs n=400's ~18.25/min)
        # while giving comparable node-count richness to RR/WS's own training
        # tiers (n=300 avg~485 nodes vs RR's 110-967, WS's 104-1826) --
        # n=150-250 were tested and found degenerate (1-26 nodes, too easy).
        # This also reduces the BA-vs-RR/WS training-tier difficulty
        # asymmetry that n=400 had (n=400 was ~4.75x/~2.1x harder than
        # RR's/WS's tiers by avg solve time -- a confound flagged as likely
        # affecting the RPB tuning results, since a harder/richer training
        # signal for BA alone, independent of any genuine cross-domain
        # robustness, could explain why ba-tuned RPB params generalized best).
        return generate_ba_instance(seed=seed, n_nodes=300, affinity=5)
    elif domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=200, degree=3)
    elif domain == "ws":
        return generate_ws_instance(seed=seed, n_nodes=200, k=20, p=0.2)
    raise ValueError(f"unknown domain {domain!r}")


class ExploreThenStrongBranchFlash:
    """Root node always uses strong branching; other nodes use it with
    `expert_probability`, otherwise fall back to cheap pseudocosts so the
    search visits states beyond what a flawless expert trajectory would reach."""

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        self.is_root_node = True

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)
        self.is_root_node = True

    def extract(self, model, done):
        if self.is_root_node:
            self.is_root_node = False
            return (self.strong_branching_function.extract(model, done), True)
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        return (self.pseudocosts_function.extract(model, done), False)


def make_env(expert_probability):
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs.set_active_features(list(range(NUM_NODE_FEATURES)))
    flash_khalil_obs.set_active_features(list(range(NUM_KHALIL_FEATURES)))
    scip_params = get_scip_params(time_limit=TRAIN_TIME_LIMIT)
    env = ecole.environment.Branching(
        observation_function={
            "Branch": ExploreThenStrongBranchFlash(expert_probability),
            "Node": flash_node_obs,
            "Khalil": flash_khalil_obs,
        },
        scip_params=scip_params,
    )
    return env


def generate_split(domain, split_name, num_samples_target, seed_start, out_dir, expert_probability=0.6,
                    file_prefix="sample", log_prefix=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = make_env(expert_probability)
    env.seed(0)
    log_tag = log_prefix if log_prefix is not None else split_name

    sample_counter = 0
    instance_counter = 0
    seed = seed_start

    while sample_counter < num_samples_target:
        instance = make_train_instance(domain, seed)
        observation, action_set, _, done, _ = env.reset(instance)

        while not done and sample_counter < num_samples_target:
            scores, scores_are_expert = observation["Branch"]

            if scores_are_expert:
                variable_features = np.concatenate(
                    (observation["Node"].variable_features[action_set, :],
                     observation["Khalil"].features[action_set, :]),
                    axis=1,
                )
                variable_features = normalize(variable_features.copy(), USED_FEATURES)
                label = int(np.argmax(scores[action_set]))

                sample_path = os.path.join(out_dir, f"{file_prefix}_{sample_counter}.pkl")
                with gzip.open(sample_path, "wb") as f:
                    pickle.dump(
                        {"features": variable_features.astype(np.float32), "label": label, "seed": seed},
                        f,
                    )
                sample_counter += 1
                if sample_counter % 1000 == 0:
                    print(f"[{log_tag}] {sample_counter}/{num_samples_target} samples "
                          f"from {instance_counter + 1} instances", flush=True)

            action = action_set[scores[action_set].argmax()]
            observation, action_set, _, done, _ = env.step(action)

        instance_counter += 1
        seed += 1

    print(f"[{log_tag}] done: {sample_counter} samples from {instance_counter} instances "
          f"(seeds {seed_start}..{seed - 1})")
    return sample_counter, instance_counter


def _worker_entry(args):
    worker_id, domain, split_name, num_samples_target, seed_start, out_dir, expert_probability = args
    return generate_split(
        domain, split_name, num_samples_target, seed_start, out_dir, expert_probability,
        file_prefix=f"w{worker_id}", log_prefix=f"{domain}/{split_name}/worker{worker_id}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ba", "rr", "ws"], required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed_start", type=int, required=True)
    parser.add_argument("--expert_probability", type=float, default=0.6)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), f"data_{args.domain}", args.split)

    if args.num_workers > 1:
        SEED_BLOCK = 10_000_000
        samples_per_worker = args.num_samples // args.num_workers
        tasks = [
            (i, args.domain, args.split, samples_per_worker, args.seed_start + i * SEED_BLOCK,
             out_dir, args.expert_probability)
            for i in range(args.num_workers)
        ]
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = pool.map(_worker_entry, tasks)
        total_samples = sum(r[0] for r in results)
        total_instances = sum(r[1] for r in results)
        print(f"[{args.domain}/{args.split}] TOTAL: {total_samples} samples from {total_instances} "
              f"instances across {args.num_workers} workers")
    else:
        generate_split(args.domain, args.split, args.num_samples, args.seed_start, out_dir,
                        args.expert_probability)
