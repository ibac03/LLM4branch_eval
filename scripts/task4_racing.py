"""Task 4, cascade idea: start several (policy, random seed) runs for 50 nodes, then continue only the best one.

    python scripts/task4_racing.py --dir runs/task4_easy --cap 300 [--exclude seed,seed,...]

Reads <dir>/portfolio.csv, <dir>/features.csv (probe of orig, seed 0) and <dir>/probes/<policy>_s<seed>.csv.
A "config" is one policy with one random seed. All time numbers include the cost of the probes that
were stopped. The model that predicts the solve time from the probe is cross-validated by instance.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--cap", type=float, default=300)
p.add_argument("--exclude", default="")
p.add_argument("--order", default="orig_s0,orig_s1,orig_retuned_s0,degree_s0,orig_s2,orig_retuned_s1,degree_s1")
a = p.parse_args()
D = Path(a.dir)
NAVY, ORANGE, GREEN, GREY = "#3f5a8a", "#d9822b", "#2e8b57", "#888888"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

raw = pd.read_csv(D / "portfolio.csv")
raw["cfg"] = raw.policy + "_s" + raw.env_seed.astype(str)
raw["time"] = np.where(raw.status == "optimal", raw.time, a.cap).clip(max=a.cap)
T = raw.pivot(index="seed", columns="cfg", values="time")
probes = {"orig_s0": pd.read_csv(D / "features.csv").set_index("seed")}
for f in sorted((D / "probes").glob("*.csv")):
    probes[f.stem] = pd.read_csv(f).set_index("seed")
cfgs = [c for c in a.order.split(",") if c in probes and c in T.columns]
excl = [int(x) for x in a.exclude.split(",") if x]
idx = T[cfgs].dropna().index.difference(excl)
for c in cfgs:
    idx = idx.intersection(probes[c].dropna(subset=["k50_time"]).index)
T = T.loc[idx, cfgs]
n = len(T)
PF = ["r_gap", "r_n_cands", "r_lp_iters", "r_time", "k5_gap", "k10_gap", "k25_gap", "k50_gap", "k10_dual_move", "k25_dual_move",
      "k50_dual_move", "k25_open_nodes", "k50_open_nodes", "k50_depth", "k50_lp_iters", "k50_time", "k50_n_sols",
      "k50_cands_mean", "k50_cands_last", "k50_margin_mean", "finished_in_probe"]
cost = pd.DataFrame({c: probes[c].loc[idx, "k50_time"] for c in cfgs})          # time of one 50-node probe
cost = cost.where(cost < T, T)                                                   # a probe never costs more than the full solve
gap50 = pd.DataFrame({c: probes[c].loc[idx, "k50_gap"] for c in cfgs})

# one row per (instance, config): predict log solve time from the probe of that config
long = pd.concat([probes[c].loc[idx, PF].assign(cfg=c, seed=idx, policy=c.rsplit("_s", 1)[0], y=np.log(T[c].to_numpy())) for c in cfgs])
X = pd.concat([long[PF], pd.get_dummies(long.policy, dtype=float)], axis=1)
pred_long = np.zeros(len(long))
for tr, te in GroupKFold(5).split(X, long.y, groups=long.seed):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X.iloc[tr], long.y.iloc[tr])
    pred_long[te] = m.predict(X.iloc[te])
long["pred"] = pred_long
P = long.pivot(index="seed", columns="cfg", values="pred").loc[idx, cfgs]

out = {"n_instances": n, "configs_in_order": cfgs, "mean_probe_time_s": float(cost.mean().mean()),
       "probe_time_%_of_mean_solve": float(100 * cost.mean().mean() / T[cfgs[0]].mean())}
# does the probe tell which of two runs of the SAME instance is faster?
within = {}
for c in cfgs[1:]:
    dt = np.log(T[c] / T[cfgs[0]])
    within[c] = {"spearman_gap50_diff_vs_time_ratio": spearmanr(gap50[c] - gap50[cfgs[0]], dt)[0],
                 "spearman_model_diff_vs_time_ratio": spearmanr(P[c] - P[cfgs[0]], dt)[0],
                 "model_picks_the_faster_%": float(100 * ((P[c] < P[cfgs[0]]) == (T[c] < T[cfgs[0]])).mean())}
out["within_instance_signal_vs_" + cfgs[0]] = within
out["across_instances_spearman_pred_vs_time"] = float(spearmanr(long.pred, long.y)[0])

hard = T[cfgs[0]].rank(ascending=False, method="first") <= round(0.1 * n)
arr_T, arr_C, arr_P, arr_G = T.to_numpy(), cost.to_numpy(), P.to_numpy(), gap50.to_numpy()
rows = []


def add(name, time, k, probed):
    rows.append({"strategy": name, "configs": k, "mean_time_s": time.mean(), "mean_time_slowest_10%_s": time[hard.to_numpy()].mean(),
                 "vs_original_%": 100 * (time.mean() / arr_T[:, 0].mean() - 1), "mean_probes_started": probed})


def race(score, k, flagged=None):
    """Probe the first k configs (only on flagged instances), continue the one with the lowest score."""
    sc = score[:, :k].copy()
    if flagged is not None:
        sc[~flagged, 1:] = np.inf
    pick = sc.argmin(axis=1)
    used = np.isfinite(sc)
    wasted = (arr_C[:, :k] * used).sum(axis=1) - arr_C[np.arange(n), pick]
    return arr_T[np.arange(n), pick] + wasted, used.sum(axis=1).mean()


add("original policy, one run", arr_T[:, 0], 1, 1.0)
for k in range(2, len(cfgs) + 1):
    add(f"oracle of {k} runs, no probe cost", arr_T[:, :k].min(axis=1), k, k)
    t, u = race(arr_G, k)
    add(f"race {k}: lowest gap after 50 nodes", t, k, u)
    t, u = race(arr_P, k)
    add(f"race {k}: model", t, k, u)
    for q in (0.1, 0.2, 0.3, 0.5):
        flagged = arr_P[:, 0] >= np.quantile(arr_P[:, 0], 1 - q)
        t, u = race(arr_P, k, flagged)
        add(f"cascade {k}: race only the {int(q * 100)}% predicted-hard, model", t, k, u)
res = pd.DataFrame(rows)
res.to_csv(D / "racing_results.csv", index=False)
best = res[~res.strategy.str.startswith("oracle")].sort_values("mean_time_s").head(8)
print(res[res.strategy.str.contains("oracle|original")].round(2).to_string(index=False))
print(best.round(2).to_string(index=False))
out["best_strategy"] = best.iloc[0].to_dict()
(D / "racing_summary.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ks = range(2, len(cfgs) + 1)
base = arr_T[:, 0].mean()
sel = lambda pat: [res[(res.configs == k) & res.strategy.str.contains(pat)].mean_time_s.min() for k in ks]
ax.axhline(base, color=GREY, ls=":", lw=1.2)
ax.text(2, base * 1.01, "original policy, one run", fontsize=8, color=GREY)
ax.plot(ks, sel("oracle"), "o-", color=GREEN, label="oracle (best run, no probe cost)")
ax.plot(ks, sel("race .*: model"), "s-", color=ORANGE, label="race all runs for 50 nodes, model picks")
ax.plot(ks, sel("cascade"), "^-", color=NAVY, label="cascade: race only predicted-hard instances")
ax.set_xlabel("number of (policy, seed) runs available")
ax.set_ylabel(f"mean solve time (s), {n} instances, probe cost included")
ax.set_title("Probe several runs for 50 nodes, continue the best one", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(D / "racing.png", dpi=200)
fig.savefig(D / "racing.pdf")
