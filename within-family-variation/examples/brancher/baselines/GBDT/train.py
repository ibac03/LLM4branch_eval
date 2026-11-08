"""
Fit a LightGBM lambdarank ranker on one domain's imitation-learning data
(same 91-dim NodeBipartiteFlash+Khalil2016Flash feature space as the MLP
baseline). Unlike MLP/GNN, there's no epoch loop or checkpoint selection --
a single fit, in the LightGBM (X, y, group) ranking format: X concatenates
all candidates across all nodes, y is 1 for the expert's chosen candidate
and 0 for the rest, group holds each node's candidate-set size so the
ranker knows where one node's candidates end and the next begins.
"""
import argparse
import glob
import gzip
import os
import pickle

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(__file__)


def load_split(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not files:
        raise ValueError(f"No sample files found in {data_dir}")
    X_list, y_list, groups = [], [], []
    for fp in files:
        with gzip.open(fp, "rb") as f:
            d = pickle.load(f)
        feats = d["features"]
        label = d["label"]
        n = feats.shape[0]
        y = np.zeros(n, dtype=np.int32)
        y[label] = 1
        X_list.append(feats)
        y_list.append(y)
        groups.append(n)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ba", "rr", "ws"], required=True)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--model_out", default=None)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_dir = args.train_dir or os.path.join(HERE, f"data_{args.domain}", "train")
    val_dir = args.val_dir or os.path.join(HERE, f"data_{args.domain}", "val")
    model_out = args.model_out or os.path.join(HERE, f"gbdt_{args.domain}.txt")

    print(f"Loading train data from {train_dir}...", flush=True)
    X_train, y_train, group_train = load_split(train_dir)
    print(f"Loading val data from {val_dir}...", flush=True)
    X_val, y_val, group_val = load_split(val_dir)
    print(f"train: {X_train.shape[0]} rows / {len(group_train)} groups, "
          f"val: {X_val.shape[0]} rows / {len(group_val)} groups", flush=True)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        verbosity=-1,
    )
    ranker.fit(
        X_train, y_train, group=group_train,
        eval_set=[(X_val, y_val)], eval_group=[group_val],
        eval_at=[1],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)],
    )

    ranker.booster_.save_model(model_out)
    print(f"Saved: {model_out} (best_iteration={ranker.best_iteration_})", flush=True)
