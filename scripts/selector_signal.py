"""How well does the early probe predict the hard set? (official easy run + probe_easy.json)

    python scripts/selector_signal.py results/tasks1-3_easy/per_instance_easy.csv results/task4_pilot/probe_easy.json results/task4_pilot/specialist_split.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(sys.argv[1]).merge(pd.DataFrame(json.load(open(sys.argv[2]))), on="seed")
split = json.load(open(sys.argv[3]))
k = int(round(0.1 * len(df)))
df["worst_time"] = df["policy_time"].rank(ascending=False, method="first") <= k
df["worst_ratio"] = df["ratio"].rank(ascending=False, method="first") <= k
df["root_gap"] = (df["root_dual"] - df["root_primal"]).abs() / df["root_primal"].abs().clip(lower=1)
df["nps_50"] = 49 / (df["time_50"] - df["root_time"]).clip(lower=1e-3)
feats = ["gap_1", "gap_10", "gap_25", "gap_50", "root_gap", "root_time", "root_ncands", "nps_50"]


def cm(y, p):
    return {"caught": int((y & p).sum()), "missed": int((y & ~p).sum()), "false_alarm": int((~y & p).sum()),
            "cleared": int((~y & ~p).sum())}


out = {}
for target in ["worst_time", "worst_ratio"]:
    y = df[target].to_numpy()
    res = {"mean_worst_vs_rest": {f: [round(df.loc[y, f].mean(), 4), round(df.loc[~y, f].mean(), 4)] for f in feats},
           "spearman_with_policy_time": {f: round(df[[f, "policy_time"]].corr("spearman").iloc[0, 1], 3) for f in feats}}
    pred = (df["gap_50"].rank(ascending=False, method="first") <= k).to_numpy()
    res["gap50_top10"] = cm(y, pred)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    prob = cross_val_predict(model, df[feats].fillna(0).to_numpy(), y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    res["logreg_cv_top10"] = cm(y, pd.Series(prob).rank(ascending=False, method="first").to_numpy() <= k)
    out[target] = res
print(json.dumps(out, indent=1))
