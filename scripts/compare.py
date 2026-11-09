"""Compare our official-seed run with Hardik's 1300 results (the PDF data) -> runs/comparison/.

    python scripts/compare.py

Reads runs/medium_official/per_instance.csv (from analyze.py) and Hardik's
worst_decile_analysis_results.json. The instances are different (other seeds), so this compares
distributions, not single instances. Our times are cut at 900 s, as in Hardik's runs.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
HARDIK_JSON = ROOT / "within-family-variation/examples/brancher/experiments/structural_generalization/worst_decile_analysis_results.json"
CAP = 900

ours = pd.read_csv(ROOT / "runs/medium_official/per_instance.csv")
ours["policy_time"] = ours.policy_time.clip(upper=CAP)
ours["rpb_time"] = ours.rpb_time.clip(upper=CAP)
ours["ratio"] = ours.policy_time / ours.rpb_time
his = pd.DataFrame(json.load(open(HARDIK_JSON)))


def stats(d):
    k = int(round(0.1 * len(d)))
    worst = d.ratio.rank(ascending=False, method="first") <= k
    slow = d.policy_time.rank(ascending=False, method="first") <= k
    q = d.ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n_instances": int(len(d)),
        "ratio_median": q[0.5], "ratio_mean": d.ratio.mean(),
        "ratio_q10": q[0.1], "ratio_q25": q[0.25], "ratio_q75": q[0.75], "ratio_q90": q[0.9],
        "worst_decile_ratio_cutoff": d.ratio[worst].min(),
        "n_ratio_above_1": int((d.ratio > 1).sum()),
        "policy_mean_s": d.policy_time.mean(), "rpb_mean_s": d.rpb_time.mean(),
        "policy_median_s": d.policy_time.median(), "rpb_median_s": d.rpb_time.median(),
        "n_time_outs_900s": int((d.policy_time >= 0.99 * CAP).sum() + (d.rpb_time >= 0.99 * CAP).sum()),
        "slowest_decile_share_of_policy_time": d.policy_time[slow].sum() / d.policy_time.sum(),
        "worst_ratio_vs_slowest_time_decile_overlap": int((worst & slow).sum()),
    }


s_ours, s_his = stats(ours), stats(his)
unfinished = json.loads((ROOT / "runs/medium_official/summary.json").read_text()).get("unfinished_seeds", [])
s_ours["n_unfinished (stopped, not in the numbers)"] = len(unfinished)
s_his["n_unfinished (stopped, not in the numbers)"] = 0
ks = ks_2samp(ours.ratio, his.ratio)
summary = {
    "ours_official_seeds": s_ours,
    "hardik_pdf": s_his,
    "ratio_distribution_test": {"ks_statistic": ks.statistic, "ks_p_value": ks.pvalue},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))

rows = [(k, s_ours[k], s_his[k]) for k in s_ours]
pd.DataFrame(rows, columns=["metric", "ours (official seeds)", "Hardik (PDF)"]).to_csv(OUT / "summary_table.csv", index=False)

NAVY, ORANGE, RED = "#3f5a8a", "#d9822b", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
bins = np.arange(0, max(1.3, ours.ratio.max(), his.ratio.max()) + 0.05, 0.05)
ax1.hist(his.ratio, bins=bins, color=ORANGE, alpha=0.55, label=f"Hardik / PDF (median {s_his['ratio_median']:.2f})")
ax1.hist(ours.ratio, bins=bins, histtype="step", lw=1.8, color=NAVY, label=f"ours, official seeds (median {s_ours['ratio_median']:.2f})")
ax1.axvline(1.0, color=RED, ls="--", lw=1)
ax1.set_xlabel("ratio = policy time / RPB time")
ax1.set_ylabel("number of instances")
ax1.set_title("Ratio histogram, 1300 BA instances (n=1000) each", fontsize=10)
ax1.legend(fontsize=8, frameon=False)

for d, col, lab in [(his, ORANGE, "Hardik / PDF"), (ours, NAVY, "ours, official seeds")]:
    x = np.sort(d.ratio)
    ax2.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=1.8, label=lab)
ax2.axvline(1.0, color=RED, ls="--", lw=1)
ax2.axhline(0.9, color="#999999", ls=":", lw=1)
ax2.set_xlabel("ratio = policy time / RPB time")
ax2.set_ylabel("fraction of instances with ratio <= x")
ax2.set_title(f"Cumulative distribution (KS statistic {ks.statistic:.3f})", fontsize=10)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "ratio_vs_pdf.pdf")
fig.savefig(OUT / "ratio_vs_pdf.png", dpi=200)
