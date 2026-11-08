"""Simulate the two-policy system on the 1300 easy instances.

Inputs: official per-instance results for Original + RPB, the specialist's official tester log,
the probe features, and the train/valid split (those 8 seeds are excluded from all reported numbers).

Selector: cross-validated logistic regression on the 50-node probe features flags the predicted-hard
10%. Predicted normal -> Original keeps solving (probe is just its first 50 nodes, no extra cost).
Predicted hard -> probe time is wasted, then the specialist solves from scratch.

    python scripts/selector_sim.py results/tasks1-3_easy/per_instance_easy.csv results/runs/specialist_easy.log \
        results/task4_pilot/probe_easy.json results/task4_pilot/specialist_split.json results/task4_specialist
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

base = pd.read_csv(sys.argv[1])
pat = re.compile(r"Seed: (\d+), Method: (\[.*?\]), Proc Time: (\[.*?\]), Node: (\[.*?\])")
spec = []
for line in Path(sys.argv[2]).read_text().splitlines():
    m = pat.search(line)
    if m:
        meth, t, n = (ast.literal_eval(g) for g in m.groups()[1:])
        i = meth.index("evolve")
        spec.append({"seed": int(m.group(1)), "spec_time": t[i], "spec_nodes": n[i]})
df = base.merge(pd.DataFrame(spec).drop_duplicates("seed"), on="seed").merge(
    pd.DataFrame(json.load(open(sys.argv[3]))), on="seed")
split = json.load(open(sys.argv[4]))
out = Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)

k = int(round(0.1 * len(base)))
hard_seeds = set(base.nlargest(k, "policy_time").seed)
df["hard"] = df.seed.isin(hard_seeds)
df["root_gap"] = (df.root_dual - df.root_primal).abs() / df.root_primal.abs().clip(lower=1)
df["nps_50"] = 49 / (df.time_50 - df.root_time).clip(lower=1e-3)
feats = ["gap_1", "gap_10", "gap_25", "gap_50", "root_gap", "root_time", "root_ncands", "nps_50"]

used = set(split["train_seeds"]) | set(split["valid_seeds"])
ev = df[~df.seed.isin(used)].reset_index(drop=True)
y = ev.hard.to_numpy()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
prob = cross_val_predict(model, ev[feats].fillna(0).to_numpy(), y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
flag = pd.Series(prob).rank(ascending=False, method="first").to_numpy() <= int(round(0.1 * len(ev)))

probe_cost = np.minimum(ev.time_50, ev.policy_time)  # wasted only when we switch
ev["t_selector"] = np.where(flag, probe_cost + ev.spec_time, ev.policy_time)
ev["t_oracle"] = np.minimum(ev.policy_time, ev.spec_time)


def g(x, s=1.0):
    return float(np.exp(np.mean(np.log(np.asarray(x) + s))) - s)


rows = []
for name, col in [("RPB", "rpb_time"), ("Original", "policy_time"), ("Specialist everywhere", "spec_time"),
                  ("Selector (probe + switch)", "t_selector"), ("Oracle (best of the two)", "t_oracle")]:
    rows.append({"system": name, "mean_all": ev[col].mean(), "gmean_all": g(ev[col]),
                 "mean_hard": ev.loc[ev.hard, col].mean(), "mean_rest": ev.loc[~ev.hard, col].mean()})
tbl = pd.DataFrame(rows)
hard = ev[ev.hard]
summary = {
    "n_eval": len(ev), "n_hard_heldout": int(ev.hard.sum()),
    "hard_spec_faster_count": int((hard.spec_time < hard.policy_time).sum()),
    "hard_median_speedup_orig_over_spec": float(np.median(hard.policy_time / hard.spec_time)),
    "rest_median_speedup_orig_over_spec": float(np.median(ev.loc[~ev.hard, "policy_time"] / ev.loc[~ev.hard, "spec_time"])),
    "selector_caught": int((flag & y).sum()), "selector_false_alarm": int((flag & ~y).sum()),
    "selector_missed": int((~flag & y).sum()),
}
print(tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(json.dumps(summary, indent=1))
tbl.to_csv(out / "systems.csv", index=False)
ev.to_csv(out / "per_instance_with_specialist.csv", index=False)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
