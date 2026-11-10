"""Task 4 analysis: policy table, seed control, hard-instance features, router.

    python scripts/task4_analysis.py --dir runs/task4_easy --cap 300 [--exclude 1000000422,1000000770,...]

Reads <dir>/portfolio.csv (run_portfolio.py) and <dir>/features.csv (probe_features.py).
Writes tables (CSV), summary.json and plots into <dir>. A solve that did not finish counts as the cap.
All models use 5-fold cross-validation: each instance is predicted by a model that did not see it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--cap", type=float, default=300)
p.add_argument("--exclude", default="", help="seeds to leave out (used to train a policy)")
p.add_argument("--max-router-policies", type=int, default=4)
a = p.parse_args()
D = Path(a.dir)
NAVY, ORANGE, GREEN, RED, GREY = "#3f5a8a", "#d9822b", "#2e8b57", "#c0392b", "#888888"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------- load ----------
raw = pd.read_csv(D / "portfolio.csv")
raw["key"] = np.where(raw.env_seed == 0, raw.policy, raw.policy + "@seed" + raw.env_seed.astype(str))
raw["solved"] = raw.status == "optimal"
raw["time"] = np.where(raw.solved, raw.time, a.cap).clip(max=a.cap)
T = raw.pivot(index="seed", columns="key", values="time")
N = raw.pivot(index="seed", columns="key", values="nodes")
S = raw.pivot(index="seed", columns="key", values="solved")
excl = [int(x) for x in a.exclude.split(",") if x]
keep = T.dropna().index.difference(excl)
T, N, S = T.loc[keep], N.loc[keep], S.loc[keep].astype(bool)
n = len(T)
seed_keys = [k for k in T.columns if k.startswith("orig@seed")]
policies = [k for k in T.columns if "@seed" not in k]
out = {"n_instances": n, "cap_s": a.cap, "excluded_seeds": excl}


def gmean(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float) + 1))) - 1)


# ---------- 1. policy table ----------
best_time = T[policies].min(axis=1)
rows = []
for k in list(T.columns):
    both = S[k] & S["orig"]
    rows.append({
        "policy": k, "mean_time_s": T[k].mean(), "gmean_time_s": gmean(T[k]), "gmean_nodes": gmean(N[k][S[k]]),
        "not_solved": int((~S[k]).sum()),
        "faster_than_orig_%": 100 * (T[k] < T["orig"]).mean(),
        "fewer_nodes_than_orig_%": 100 * (N[k][both] < N["orig"][both]).mean(),
        "is_fastest_%": 100 * (T[k] <= best_time).mean() if k in policies else np.nan,
        "mean_time_of_best_of_orig_and_this_s": np.minimum(T["orig"], T[k]).mean(),
    })
table = pd.DataFrame(rows).sort_values("mean_time_s").reset_index(drop=True)
table.to_csv(D / "policy_table.csv", index=False)
print(table.round(2).to_string(index=False))

# ---------- 2. seed control: the same policy with a different random seed ----------
def spread(x, y):
    r = np.abs(np.log2((x + 0.5) / (y + 0.5)))
    return {"spearman": spearmanr(x, y)[0], "median_factor": float(2 ** r.median()), "p90_factor": float(2 ** r.quantile(0.9))}


if seed_keys:
    hard1 = T[seed_keys[0]].rank(ascending=False, method="first") <= round(0.1 * n)
    hard0 = T["orig"].rank(ascending=False, method="first") <= round(0.1 * n)
    out["slowest_decile_same_with_other_seed"] = f"{int((hard0 & hard1).sum())} of {int(hard0.sum())}"
    sc = {"time, orig vs " + k: spread(T["orig"], T[k]) for k in seed_keys}
    sc.update({"nodes, orig vs " + k: spread(N["orig"], N[k]) for k in seed_keys})
    sc["mean_time_orig_s"] = T["orig"].mean()
    sc["mean_time_best_of_2_seeds_s"] = np.minimum(T["orig"], T[seed_keys[0]]).mean()
    sc["mean_time_best_of_all_seeds_s"] = T[["orig"] + seed_keys].min(axis=1).mean()
    out["seed_control"] = sc

# ---------- 3. which features find the hard instances ----------
F = pd.read_csv(D / "features.csv").set_index("seed")
F = F.drop(columns=[c for c in ("error", "nodes_reached") if c in F.columns]).reindex(T.index)
F = F.loc[:, F.nunique() > 1]
y = np.log(T["orig"])
hard = T["orig"].rank(ascending=False, method="first") <= round(0.1 * n)


def group_of(c):
    return {"g": "graph", "r": "root"}.get(c.split("_")[0], c.split("_")[0]) if c != "finished_in_probe" else "k50"


STAGES = ["graph", "root", "k1", "k5", "k10", "k25", "k50"]
STAGE_LABEL = {"graph": "graph only", "root": "+ root node", "k1": "1 node", "k5": "5 nodes", "k10": "10 nodes", "k25": "25 nodes", "k50": "50 nodes"}
COST_COL = {"graph": None, "root": "r_time", "k1": "k1_time", "k5": "k5_time", "k10": "k10_time", "k25": "k25_time", "k50": "k50_time"}
fr = []
for c in F.columns:
    x = F[c].fillna(F[c].median())
    auc = roc_auc_score(hard, x)
    fr.append({"feature": c, "group": group_of(c), "spearman_with_solve_time": spearmanr(x, y)[0], "auc_hard_decile": max(auc, 1 - auc)})
fr = pd.DataFrame(fr).sort_values("spearman_with_solve_time", key=np.abs, ascending=False).reset_index(drop=True)
fr.to_csv(D / "feature_ranking.csv", index=False)

cv = KFold(5, shuffle=True, random_state=0)


def cols_upto(stage):
    ok = STAGES[: STAGES.index(stage) + 1]
    return [c for c in F.columns if group_of(c) in ok]


def fit_predict(X, target):
    return cross_val_predict(HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0), X, target, cv=cv)


hs = []
for st in STAGES:
    X = F[cols_upto(st)]
    if X.shape[1] == 0:
        continue
    pred = fit_predict(X, y)
    flag = pd.Series(pred, index=y.index).rank(ascending=False, method="first") <= hard.sum()
    cost = F[COST_COL[st]].mean() if COST_COL[st] else 0.0
    hs.append({"features": STAGE_LABEL[st], "n_features": X.shape[1], "spearman_pred_vs_time": spearmanr(pred, y)[0],
               "auc_hard_decile": roc_auc_score(hard, pred), "hard_found_of_%d" % hard.sum(): int((flag & hard).sum()),
               "feature_cost_s": cost, "feature_cost_%_of_mean_solve": 100 * cost / T["orig"].mean()})
hs = pd.DataFrame(hs)
hs.to_csv(D / "hard_prediction_by_stage.csv", index=False)
print(hs.round(3).to_string(index=False))
ceiling = spearmanr(T["orig"], T[seed_keys[0]])[0] if seed_keys else None
out["hard_prediction"] = {"spearman_ceiling_from_seed_noise": ceiling}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
top = fr.head(15).iloc[::-1]
gcol = {"graph": GREY, "root": ORANGE}
ax1.barh(top.feature, top.spearman_with_solve_time.abs(), color=[gcol.get(g, NAVY) for g in top.group])
ax1.set_xlabel("|Spearman correlation| with the solve time of the original policy")
ax1.set_title("Best single features (grey: graph, orange: root, blue: first nodes)", fontsize=10)
xs = range(len(hs))
ax2.plot(xs, hs.spearman_pred_vs_time, "o-", color=NAVY, label="Spearman: predicted vs real solve time")
ax2.plot(xs, hs.iloc[:, 4] / hard.sum(), "s-", color=ORANGE, label="part of the slowest 10% that the model finds")
if ceiling:
    ax2.axhline(ceiling, color=GREY, ls=":", lw=1.2)
    ax2.text(0, ceiling + 0.01, "limit from random-seed noise", fontsize=8, color=GREY)
ax2.set_xticks(list(xs))
ax2.set_xticklabels(hs.features, rotation=30, ha="right")
ax2.set_ylim(0, 1)
ax2.set_title("Prediction of hard instances vs how long we look", fontsize=10)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(D / "hard_features.png", dpi=200)
fig.savefig(D / "hard_features.pdf")

# ---------- 4. router ----------
# candidate policies: greedy, each step adds the policy that makes the oracle (best per instance) fastest
chosen = ["orig"]
while len(chosen) < min(a.max_router_policies, len(policies)):
    gains = {k: T[chosen + [k]].min(axis=1).mean() for k in policies if k not in chosen}
    k = min(gains, key=gains.get)
    if gains[k] > 0.995 * T[chosen].min(axis=1).mean():
        break
    chosen.append(k)
single_best = T[policies].mean().idxmin()
systems = {"RPB": T["rpb"] if "rpb" in T else None, "original policy": T["orig"], f"best single policy ({single_best})": T[single_best]}
rt = []
for st in STAGES:
    X = F[cols_upto(st)]
    if X.shape[1] == 0 or len(chosen) < 2:
        continue
    pred = pd.DataFrame({k: fit_predict(X, np.log(T[k])) for k in chosen}, index=T.index)
    cost = F[COST_COL[st]].fillna(0) if COST_COL[st] else 0.0
    est = np.exp(pred)  # predicted time; a switch away from the original also pays the feature cost
    est.loc[:, est.columns != "orig"] = est.loc[:, est.columns != "orig"].add(cost, axis=0) if COST_COL[st] else est.loc[:, est.columns != "orig"]
    pick = est.idxmin(axis=1)
    time = pd.Series(T.to_numpy()[np.arange(n), T.columns.get_indexer(pick)], index=T.index) + np.where(pick != "orig", cost, 0.0)
    systems[f"router, {STAGE_LABEL[st]}"] = time
    rt.append({"features": STAGE_LABEL[st], "switch_%": 100 * (pick != "orig").mean(),
               "picked_the_fastest_%": 100 * (pick == T[chosen].idxmin(axis=1)).mean()})
systems[f"oracle over {len(chosen)} policies"] = T[chosen].min(axis=1)
systems[f"oracle over all {len(policies)} policies"] = T[policies].min(axis=1)
if seed_keys:
    systems[f"oracle over {len(seed_keys) + 1} random seeds of the original"] = T[["orig"] + seed_keys].min(axis=1)
res = pd.DataFrame([{"system": k, "mean_time_s": v.mean(), "gmean_time_s": gmean(v), "mean_time_slowest_10%_s": v[hard].mean(),
                     "vs_original_%": 100 * (v.mean() / T["orig"].mean() - 1)} for k, v in systems.items() if v is not None])
res.to_csv(D / "router_results.csv", index=False)
pd.DataFrame(rt).to_csv(D / "router_choices.csv", index=False)
print("router policies:", chosen)
print(res.round(2).to_string(index=False))
print(pd.DataFrame(rt).round(1).to_string(index=False))
out["router_policies"] = chosen

r2 = res[~res.system.str.contains("router, (1 node|5 nodes|25 nodes)")].iloc[::-1]
fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(r2) + 1.2))
col = [GREEN if s.startswith("oracle") else ORANGE if s.startswith("router") else NAVY for s in r2.system]
ax.barh(r2.system, r2.mean_time_s, color=col)
for i, v in enumerate(r2.mean_time_s):
    ax.text(v, i, f" {v:.1f} s", va="center", fontsize=8)
ax.axvline(T["orig"].mean(), color=GREY, ls=":", lw=1)
ax.set_xlabel(f"mean solve time (s), {n} instances")
ax.set_title("Router against single policies and the oracle", fontsize=10)
fig.tight_layout()
fig.savefig(D / "router.png", dpi=200)
fig.savefig(D / "router.pdf")

(D / "summary.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))
