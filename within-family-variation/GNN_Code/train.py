"""
Train the GNN branching baseline (Gasse et al. 2019 bipartite GCNN) via
imitation learning against strong-branching labels.

Mirrors MLP/train.py's overall structure (per-node cross-entropy loss over
each node's candidate set, one checkpoint per epoch, final selection by
actual solving time happens separately in select_checkpoint.py) but adapted
for the bipartite-graph data format:

- Each sample stores the FULL node's variable_features/row_features (not
  just candidates) plus a `graph_id` pointing at a `graph_{id}.pkl` in the
  same instance directory (edge structure, deduplicated since it's constant
  across most nodes of one B&B tree -- see generate_data_ba.py's docstring).
- A batch concatenates several (sample, graph) pairs into ONE big bipartite
  graph with block-diagonal edge structure: each sample's variable/
  constraint/candidate indices are offset by the running totals already in
  the batch before concatenating, so cross-sample edges never accidentally
  connect (standard GNN-branching batching, same idea as block-diagonal
  adjacency batching in any GNN library, done by hand here since model.py
  deliberately avoids a graph-library dependency).
- model.py's forward() scores ALL variables in the batch; the loss only
  needs the candidate subset, so scores are gathered at the (now-global)
  candidate indices before the same per-node segment slicing MLP's
  batch_loss uses.
"""
import argparse
import glob
import gzip
import os
import pickle
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import GNNPolicy

HERE = os.path.dirname(__file__)


class BranchingSampleDataset(Dataset):
    """Each item is one node's raw (ungathered) sample -- collate() does the
    batching/offsetting so __getitem__ stays a cheap single-file read."""

    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*", "sample_*.pkl")))
        if not self.files:
            raise ValueError(f"No sample files found under {data_dir}/*/sample_*.pkl")
        self._graph_cache = {}

    def __len__(self):
        return len(self.files)

    def _load_graph(self, instance_dir, graph_id):
        key = (instance_dir, graph_id)
        if key not in self._graph_cache:
            graph_path = os.path.join(instance_dir, f"graph_{graph_id}.pkl")
            with gzip.open(graph_path, "rb") as f:
                self._graph_cache[key] = pickle.load(f)
        return self._graph_cache[key]

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        instance_dir = os.path.dirname(sample_path)
        with gzip.open(sample_path, "rb") as f:
            s = pickle.load(f)
        g = self._load_graph(instance_dir, s["graph_id"])
        return {
            "variable_features": torch.from_numpy(s["variable_features"]),
            "row_features": torch.from_numpy(s["row_features"]),
            "edge_indices": torch.from_numpy(g["edge_indices"].astype(np.int64)),
            "edge_values": torch.from_numpy(g["edge_values"]),
            "candidates": torch.from_numpy(s["candidates"].astype(np.int64)),
            "label": s["label"],
        }


def collate(batch):
    """Block-diagonal batching: offset each sample's variable/constraint/
    candidate indices by the running totals already placed in the batch,
    then concatenate everything into one big bipartite graph. Returns the
    concatenated tensors plus per-sample candidate-count sizes and local
    labels, exactly like MLP/train.py's collate for the loss slicing."""
    var_feats, row_feats, edge_idx_list, edge_val_list, cand_list, sizes, labels = [], [], [], [], [], [], []
    var_offset, cons_offset = 0, 0
    for item in batch:
        n_vars = item["variable_features"].shape[0]
        n_cons = item["row_features"].shape[0]

        var_feats.append(item["variable_features"])
        row_feats.append(item["row_features"])

        e = item["edge_indices"].clone()
        e[0] += cons_offset
        e[1] += var_offset
        edge_idx_list.append(e)
        edge_val_list.append(item["edge_values"])

        cand_list.append(item["candidates"] + var_offset)
        sizes.append(item["candidates"].shape[0])
        labels.append(item["label"])

        var_offset += n_vars
        cons_offset += n_cons

    return {
        "variable_features": torch.cat(var_feats, dim=0),
        "row_features": torch.cat(row_feats, dim=0),
        "edge_indices": torch.cat(edge_idx_list, dim=1),
        "edge_values": torch.cat(edge_val_list, dim=0),
        "candidates": torch.cat(cand_list, dim=0),
        "sizes": sizes,
        "labels": labels,
    }


def batch_loss(model, batch, device):
    all_scores = model(
        batch["variable_features"].to(device),
        batch["row_features"].to(device),
        batch["edge_indices"].to(device),
        batch["edge_values"].to(device),
    ).squeeze(-1)
    cand_scores = all_scores[batch["candidates"].to(device)]

    loss = 0.0
    correct = 0
    idx = 0
    for size, label in zip(batch["sizes"], batch["labels"]):
        seg_scores = cand_scores[idx: idx + size]
        loss = loss + F.cross_entropy(seg_scores.unsqueeze(0), torch.tensor([label], device=device))
        correct += int(seg_scores.argmax().item() == label)
        idx += size
    return loss / len(batch["sizes"]), correct / len(batch["sizes"])


def run_epoch(model, loader, device, optimizer=None, log_every=200, log_tag=""):
    # Periodic progress logging: an earlier run of this loop went 8.5+ hours
    # with ZERO visibility (only a per-EPOCH print existed) and had to be
    # killed blind -- always log per-batch pace so a stall/slowdown is
    # visible in the log itself rather than discovered after waiting hours.
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_acc, num_batches = 0.0, 0.0, 0
    num_total_batches = len(loader)
    t_start = time.time()
    t_last = t_start
    for batch in loader:
        with torch.set_grad_enabled(train_mode):
            loss, acc = batch_loss(model, batch, device)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        if num_batches % log_every == 0 or num_batches == num_total_batches:
            now = time.time()
            recent_rate = log_every / max(now - t_last, 1e-9)
            overall_rate = num_batches / max(now - t_start, 1e-9)
            eta_s = (num_total_batches - num_batches) / max(overall_rate, 1e-9)
            print(f"  [{log_tag}] batch {num_batches}/{num_total_batches} "
                  f"loss={total_loss / num_batches:.4f} acc={total_acc / num_batches:.4f} "
                  f"recent={recent_rate:.2f}batch/s overall={overall_rate:.2f}batch/s "
                  f"eta={eta_s / 60:.1f}min", flush=True)
            t_last = now
    return total_loss / num_batches, total_acc / num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ba", "rr", "ws"], required=True,
                         help="Which structural regime's data/checkpoints to use "
                              "(sets default --train_dir/--val_dir/--checkpoint_dir "
                              "to data_{domain}/train, data_{domain}/val, checkpoints_{domain}).")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    # NOTE: on the original run's hardware (Apple Silicon), MPS was
    # observed to be dramatically slower and prone to long unexplained
    # stalls under system memory pressure compared to plain CPU -- CPU was
    # actually faster and stable end-to-end. Recommend starting with
    # --device cpu unless you've separately verified MPS/CUDA behaves well
    # on your machine. See README.md's "Known issues" section.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0,
                         help="torch/shuffle seed, for reproducibility across runs/machines "
                              "(training is NOT deterministic without this).")
    args = parser.parse_args()

    args.train_dir = args.train_dir or os.path.join(HERE, f"data_{args.domain}", "train")
    args.val_dir = args.val_dir or os.path.join(HERE, f"data_{args.domain}", "val")
    args.checkpoint_dir = args.checkpoint_dir or os.path.join(HERE, f"checkpoints_{args.domain}")

    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds = BranchingSampleDataset(args.train_dir)
    val_ds = BranchingSampleDataset(args.val_dir)
    # num_workers=0: samples are read directly (cheap decompress+unpickle),
    # and the per-Dataset graph cache is only useful within one process --
    # keeping DataLoader workers at 0 avoids duplicating/losing that cache.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device}, train_samples={len(train_ds)}, val_samples={len(val_ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, log_tag=f"epoch{epoch}/train")
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None, log_tag=f"epoch{epoch}/val")

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, ckpt_path)

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}", flush=True)
