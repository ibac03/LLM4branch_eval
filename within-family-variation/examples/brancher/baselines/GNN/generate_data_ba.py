"""
Generate imitation-learning training data for the GNN branching baseline
(Gasse et al. 2019 bipartite-graph representation), on BA instances -- the
mixed-model-training ablation against the LLM4Branch/Evolved-* portfolio
(see portfolio.md section 6).

Uses n_nodes=400,affinity=5 (not indset/easy's 750,4) -- the same
"calibrated tractable BA training tier" already established and reused
throughout this project for BO retuning (tune_evolved_rr_on_ba.py,
tune_evolved_ws_on_ba.py). A first attempt at indset/easy (750,4) with no
time cap took 2.5+ minutes without even finishing 20 samples (strong
branching, the data-gen expert, evaluates every fractional candidate's LP
relaxation per node -- far more expensive than any branching rule used
elsewhere in this project) -- extrapolated to 150-250+ hours for the full
100k+20k target, so the user chose to shrink the instance size rather than
cap time or cut sample count. Also applying the same 180s-per-episode SCIP
time cap this tier has always used elsewhere (never run uncapped in this
project), since an uncapped 3600s default risked the same runaway-episode
problem even at this smaller size.

Storage is deduplicated per instance: the edge structure (constraint
coefficient matrix) never changes across nodes of the same B&B tree, so it's
saved once per instance in `graph.pkl`, while the per-node dynamic pieces
(variable_features, row_features -- both node-state-dependent, e.g. LP
fractional values and reduced costs -- plus the candidate set and expert
label) are saved one file per sample. Without this, each sample would need
its own ~700KB copy of the (unchanged) edge data.

Usage:
    python generate_data.py --split train --num_samples 100000 --seed_start 0 --num_workers 6
    python generate_data.py --split val   --num_samples 20000  --seed_start 100000000 --num_workers 6
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import get_scip_params

BA_N_NODES = 400
BA_AFFINITY = 5
BA_TIME_LIMIT = 180


def make_ba_instance(seed):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=BA_N_NODES, affinity=BA_AFFINITY,
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


class ExploreThenStrongBranch:
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
    scip_params = get_scip_params(time_limit=BA_TIME_LIMIT)
    env = ecole.environment.Branching(
        observation_function={
            "Branch": ExploreThenStrongBranch(expert_probability),
            "Node": ecole.observation.NodeBipartite(),
        },
        scip_params=scip_params,
    )
    return env


def generate_split(split_name, num_samples_target, seed_start, out_dir, expert_probability=0.6,
                    instance_prefix="i", log_prefix=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = make_env(expert_probability)
    env.seed(0)
    log_tag = log_prefix if log_prefix is not None else split_name

    sample_counter = 0
    instance_counter = 0
    seed = seed_start

    while sample_counter < num_samples_target:
        instance = make_ba_instance(seed)
        observation, action_set, _, done, _ = env.reset(instance)

        instance_dir = os.path.join(out_dir, f"{instance_prefix}{instance_counter}")
        cached_graphs = []  # list of (edge_indices, edge_values, graph_id) seen this episode
        local_sample_idx = 0
        next_graph_id = 0

        while not done and sample_counter < num_samples_target:
            scores, scores_are_expert = observation["Branch"]

            if scores_are_expert:
                node_obs = observation["Node"]
                edge_indices = node_obs.edge_features.indices.astype(np.int32)
                edge_values = node_obs.edge_features.values.astype(np.float32)

                graph_id = None
                for cached_indices, cached_values, cached_id in cached_graphs:
                    if (edge_indices.shape == cached_indices.shape
                            and np.array_equal(edge_indices, cached_indices)
                            and np.array_equal(edge_values, cached_values)):
                        graph_id = cached_id
                        break

                if graph_id is None:
                    Path(instance_dir).mkdir(parents=True, exist_ok=True)
                    graph_id = next_graph_id
                    next_graph_id += 1
                    with gzip.open(os.path.join(instance_dir, f"graph_{graph_id}.pkl"), "wb") as f:
                        pickle.dump({
                            "edge_indices": edge_indices,
                            "edge_values": edge_values,
                            "n_vars": node_obs.variable_features.shape[0],
                            "n_cons": node_obs.row_features.shape[0],
                        }, f)
                    cached_graphs.append((edge_indices, edge_values, graph_id))

                candidates = np.array(action_set, dtype=np.int32)
                label = int(np.argmax(scores[action_set]))

                sample_path = os.path.join(instance_dir, f"sample_{local_sample_idx}.pkl")
                with gzip.open(sample_path, "wb") as f:
                    pickle.dump({
                        "variable_features": node_obs.variable_features.astype(np.float32),
                        "row_features": node_obs.row_features.astype(np.float32),
                        "candidates": candidates,
                        "label": label,
                        "graph_id": graph_id,
                    }, f)
                local_sample_idx += 1
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
    worker_id, split_name, num_samples_target, seed_start, out_dir, expert_probability = args
    return generate_split(
        split_name, num_samples_target, seed_start, out_dir, expert_probability,
        instance_prefix=f"w{worker_id}_i", log_prefix=f"{split_name}/worker{worker_id}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed_start", type=int, required=True)
    parser.add_argument("--expert_probability", type=float, default=0.6)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "data_ba", args.split)

    if args.num_workers > 1:
        SEED_BLOCK = 10_000_000
        samples_per_worker = args.num_samples // args.num_workers
        tasks = [
            (i, args.split, samples_per_worker, args.seed_start + i * SEED_BLOCK, out_dir, args.expert_probability)
            for i in range(args.num_workers)
        ]
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = pool.map(_worker_entry, tasks)
        total_samples = sum(r[0] for r in results)
        total_instances = sum(r[1] for r in results)
        print(f"[{args.split}] TOTAL: {total_samples} samples from {total_instances} instances "
              f"across {args.num_workers} workers")
    else:
        generate_split(args.split, args.num_samples, args.seed_start, out_dir, args.expert_probability)
