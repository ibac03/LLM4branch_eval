"""Tester log -> per-instance CSV, summary JSON, ratio histogram, time scatter.

    python scripts/analyze.py runs/medium_official/tester.log --title "official test seeds"
    python scripts/analyze.py runs/medium_hardik/tester.log --title "Hardik's seeds" --cap 900

Outputs go next to the log. --cap T cuts every solve time at T seconds (a solve longer than T
counts as a time-out at T), as with a SCIP time limit of T. Hardik used a 900 s limit; the
official tester uses 3600 s. Both use CPU time.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--title", default="")
p.add_argument("--cap", type=float, default=3600)
p.add_argument("--seeds", help="expected seeds START:N; seeds with no result are listed as unfinished")
a = p.parse_args()
log = Path(a.log)
out = log.parent

pat = re.compile(r"Seed: (\d+), Method: (\[.*?\]), Proc Time: (\[.*?\]), Node: (\[.*?\])")
rows = []
for line in log.read_text().splitlines():
    m = pat.search(line)
    if m:
        methods, times, nodes = (ast.literal_eval(g) for g in m.groups()[1:])
        r = {"seed": int(m.group(1))}
        for meth, t, n in zip(methods, times, nodes):
            name = {"relpscost": "rpb", "evolve": "policy"}[meth]
            r[f"{name}_time"], r[f"{name}_nodes"] = t, n
        rows.append(r)
df = pd.DataFrame(rows).drop_duplicates("seed").sort_values("seed").reset_index(drop=True)
for s in ("policy", "rpb"):
    df[f"{s}_timeout"] = df[f"{s}_time"] >= 0.99 * a.cap
    df[f"{s}_time"] = df[f"{s}_time"].clip(upper=a.cap)
df["ratio"] = df["policy_time"] / df["rpb_time"]
N, k = len(df), int(round(0.1 * len(df)))
df["worst_ratio"] = df["ratio"].rank(ascending=False, method="first") <= k
df["worst_time"] = df["policy_time"].rank(ascending=False, method="first") <= k


def gmean(x, shift=1.0):
    return float(np.exp(np.mean(np.log(np.asarray(x) + shift))) - shift)


s = {
    "title": a.title, "n_instances": N, "seeds": [int(df.seed.min()), int(df.seed.max())],
    "time_cap_s": a.cap, "decile_size": k,
    "timeouts_policy": int(df.policy_timeout.sum()), "timeouts_rpb": int(df.rpb_timeout.sum()),
    "policy_mean_s": df.policy_time.mean(), "rpb_mean_s": df.rpb_time.mean(),
    "policy_gmean_s": gmean(df.policy_time), "rpb_gmean_s": gmean(df.rpb_time),
    "policy_median_s": df.policy_time.median(), "rpb_median_s": df.rpb_time.median(),
    "policy_gmean_nodes": gmean(df.policy_nodes), "rpb_gmean_nodes": gmean(df.rpb_nodes),
    "ratio_median": df.ratio.median(), "ratio_mean": df.ratio.mean(),
    "ratio_min": df.ratio.min(), "ratio_max": df.ratio.max(),
    "n_policy_slower_than_rpb": int((df.ratio > 1).sum()),
    "worst_ratio_cutoff": df.loc[df.worst_ratio, "ratio"].min(),
    "worst_time_cutoff_s": df.loc[df.worst_time, "policy_time"].min(),
    "overlap_ratio_vs_time_deciles": int((df.worst_ratio & df.worst_time).sum()),
    "slowest_decile_share_of_policy_time": df.loc[df.worst_time, "policy_time"].sum() / df.policy_time.sum(),
    "spearman_policy_vs_rpb_time": df[["policy_time", "rpb_time"]].corr("spearman").iloc[0, 1],
}
if a.seeds:
    start, n = map(int, a.seeds.split(":"))
    s["unfinished_seeds"] = sorted(set(range(start, start + n)) - set(df.seed))

NAVY, GREY, RED, ORANGE, PURPLE = "#3f5a8a", "#555555", "#c0392b", "#d9822b", "#8e44ad"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(df.ratio, bins=np.arange(0, max(1.3, df.ratio.max()) + 0.05, 0.05), color=NAVY, edgecolor="white")
top = ax.get_ylim()[1]
ax.axvline(s["worst_ratio_cutoff"], color=GREY, ls=":", lw=1.2)
ax.text(s["worst_ratio_cutoff"], top * 0.93, f" worst-decile cutoff\n ({s['worst_ratio_cutoff']:.2f})", fontsize=8, color=GREY, va="top")
ax.axvline(1.0, color=RED, ls="--", lw=1.2)
ax.text(1.0, top * 0.98, " policy = RPB", fontsize=8, color=RED, va="top")
ax.set_xlabel("ratio = policy time / RPB time")
ax.set_ylabel("number of instances")
ax.set_title(f"Policy vs RPB, {N} BA instances (n=1000), {a.title} (median {s['ratio_median']:.2f})", fontsize=10)
fig.tight_layout()
fig.savefig(out / "ratio_hist.pdf")
fig.savefig(out / "ratio_hist.png", dpi=200)

fig, ax = plt.subplots(figsize=(4.8, 4.4))
c = np.where(df.worst_ratio & df.worst_time, RED, np.where(df.worst_ratio, ORANGE, np.where(df.worst_time, PURPLE, NAVY)))
ax.scatter(df.rpb_time, df.policy_time, s=7, c=c, alpha=0.7, linewidths=0)
lo = min(df.rpb_time.min(), df.policy_time.min()) * 0.8
hi = max(df.rpb_time.max(), df.policy_time.max()) * 1.2
ax.plot([lo, hi], [lo, hi], color=RED, ls="--", lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("RPB time (s)")
ax.set_ylabel("LLM4Branch policy time (s)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=col, label=lab, markersize=5) for col, lab in
                   [(ORANGE, "worst decile by ratio"), (PURPLE, "slowest decile by time"), (RED, "both"), (NAVY, "rest")]],
          fontsize=7, loc="upper left", frameon=False)
ax.set_title(f"Per-instance CPU time, {a.title}", fontsize=10)
fig.tight_layout()
fig.savefig(out / "time_scatter.pdf")
fig.savefig(out / "time_scatter.png", dpi=200)

df.to_csv(out / "per_instance.csv", index=False)
(out / "summary.json").write_text(json.dumps(s, indent=2, default=float))
print(json.dumps(s, indent=2, default=float))
