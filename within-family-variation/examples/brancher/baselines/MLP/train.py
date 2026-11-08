"""
Train the MLP branching baseline via imitation learning against strong-branching
labels. Candidate sets vary in size per node, so a minibatch concatenates all
candidates across several nodes into one forward pass, tracking per-node
segment boundaries to compute a per-node cross-entropy loss (standard
learn2branch-style batching for this kind of variable-arity classification).

Domain-parameterized (ba/rr/ws), mirroring GBDT/GNN's --domain convention,
so the same script trains all three of the structural_generalization
crosstest's per-domain models (data_ba/train+val, data_rr/..., data_ws/...
-> checkpoints_ba/, checkpoints_rr/, checkpoints_ws/).

Two fixes vs. the original setcover-only version, both needed to run all
3 domains SAFELY IN PARALLEL as separate processes on one machine:
  - torch.manual_seed(): the original had no seeding at all, so weight
    init + minibatch shuffling were nondeterministic run-to-run (same gap
    flagged and fixed for GNN/train.py earlier in this project).
  - torch.set_num_threads(): PyTorch's default BLAS/matmul threading claims
    many cores per process; 3 domains training simultaneously without a cap
    would oversubscribe a 10-core machine. --num_threads defaults to 3
    (3 processes x 3 threads = 9, leaving the machine responsive).

Checkpoints are saved every epoch; final selection among them (by actual
solving time on held-out instances, per the paper's Appendix A protocol, not
by validation loss) happens separately in select_checkpoint.py.
"""
import argparse
import glob
import gzip
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import MLPPolicy

HERE = os.path.dirname(__file__)


class BranchingSampleDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        if not self.files:
            raise ValueError(f"No sample files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with gzip.open(self.files[idx], "rb") as f:
            d = pickle.load(f)
        # normalize() in generate_data.py can occasionally produce NaN/inf when a
        # candidate set has zero variance in some feature (safe_range hits 0) --
        # crosstest.py/select_checkpoint.py already guard against this on live
        # features via np.nan_to_num before scoring; stored samples need the same
        # guard here, since a single NaN sample poisons the whole run from epoch 1
        # (confirmed: WS training went to NaN loss starting at epoch 1 before this
        # fix, while RR/BA's data happened to have no such samples).
        features = np.nan_to_num(d["features"], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(features), d["label"]


def collate(batch):
    """Concatenate all candidates across the batch's nodes into one tensor,
    returning segment sizes and per-segment (local) labels for loss computation."""
    features, labels = zip(*batch)
    sizes = [f.shape[0] for f in features]
    return torch.cat(features, dim=0), sizes, labels


def batch_loss(model, concatenated_features, sizes, labels, device):
    scores = model(concatenated_features.to(device)).squeeze(-1)
    loss = 0.0
    correct = 0
    idx = 0
    for size, label in zip(sizes, labels):
        seg_scores = scores[idx: idx + size]
        loss = loss + F.cross_entropy(seg_scores.unsqueeze(0), torch.tensor([label], device=device))
        correct += int(seg_scores.argmax().item() == label)
        idx += size
    return loss / len(sizes), correct / len(sizes)


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_acc, num_batches = 0.0, 0.0, 0
    for concatenated_features, sizes, labels in loader:
        with torch.set_grad_enabled(train_mode):
            loss, acc = batch_loss(model, concatenated_features, sizes, labels, device)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ba", "rr", "ws"], required=True)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu",
                         help="cpu (not mps) is the safe default for running 3 domains in "
                              "parallel -- mps would serialize them on the one GPU anyway.")
    parser.add_argument("--hidden_dims", default="128,128,64", help="comma-separated hidden layer sizes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=3,
                         help="torch.set_num_threads() cap -- keeps 3 simultaneous per-domain "
                              "training processes from oversubscribing the machine's cores.")
    args = parser.parse_args()
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    train_dir = args.train_dir or os.path.join(HERE, f"data_{args.domain}", "train")
    val_dir = args.val_dir or os.path.join(HERE, f"data_{args.domain}", "val")
    checkpoint_dir = args.checkpoint_dir or os.path.join(HERE, f"checkpoints_{args.domain}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds = BranchingSampleDataset(train_dir)
    val_ds = BranchingSampleDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = MLPPolicy(hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[{args.domain}] device={device}, num_threads={args.num_threads}, "
          f"train_samples={len(train_ds)}, val_samples={len(val_ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)

        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, ckpt_path)

        print(f"[{args.domain}] epoch {epoch:3d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}", flush=True)

    print(f"[{args.domain}] TRAIN_DONE", flush=True)
