# LLM4Branch evaluation

This repository tests the LLM4Branch branching policy for Independent Set problems on
Barabási–Albert (BA) graphs. It compares the policy with the SCIP default rule (RPB, `relpscost`).
It also compares the results with Hardik's report, *LLM4Branch Within-Family Variation*.

## Contents

| Path | Contents |
|---|---|
| `LLM4Branch/` | The official LLM4Branch code ([hzn18/LLM4Branch](https://github.com/hzn18/LLM4Branch)). `setup/install.sh` downloads it. Not in git. |
| `LLM4Branch-ecole/` | The custom Ecole of the LLM4Branch authors. `setup/install.sh` downloads and builds it. Not in git. |
| `within-family-variation/` | Hardik's code. The first commit has his code as he sent it. |
| `setup/` | The install script and the exact package versions. |
| `scripts/` | `run_tester.py` runs the official tester. `analyze.py` makes the tables and plots. `compare.py` compares with Hardik's results. |
| `runs/` | The results. Each folder has the tester log, a CSV with one row for each instance, a summary and the plots. |

## Install

Use Ubuntu 24.04 (x86_64). The script puts all files in this folder (`.conda/` has Python and SCIP).

```bash
bash setup/install.sh
source .conda/etc/profile.d/conda.sh && conda activate ecole-fork
```

## Run

The medium level has graphs with 1000 nodes. The official test set starts at seed 999999999.
Hardik used seeds 800000000 to 800001299 and a 900 s time limit.

```bash
python scripts/run_tester.py --seed 999999999 --n 1300 --log runs/medium_official/tester.log
python scripts/analyze.py runs/medium_official/tester.log --title "official test seeds"
python scripts/compare.py
```

`run_tester.py` uses cores 8 to 127 by default. Change this with `--cores`.

## Results (tasks 1 to 3)

Run: 1300 medium instances from the official test set (seeds 999999999 to 1000001298), on a
Carolina Cloud VM (AMD EPYC Genoa, 120 solves at the same time). 1293 instances finished. We
stopped 7 very slow instances by hand. `runs/medium_official/summary.json` lists their seeds.

The ratio is the policy time divided by the RPB time. A ratio below 1 means that the policy is faster.

| Metric | Ours (official seeds) | Hardik (PDF) |
|---|---|---|
| Instances | 1293 | 1300 |
| Median ratio | 0.337 | 0.327 |
| 10% / 90% ratio quantiles | 0.21 / 0.58 | 0.20 / 0.58 |
| Worst-decile cutoff (ratio) | 0.582 | 0.576 |
| Instances with ratio above 1 | 12 | 14 |
| Slowest 10% of instances: part of the total policy time | 52% | 55% |
| Mean time, policy / RPB | 91 s / 201 s | 58 s / 124 s |

Both columns use a 900 s limit (`runs/comparison/summary_table.csv`).

- The ratio distributions are almost the same. A Kolmogorov–Smirnov test does not find a
  difference (statistic 0.046, p = 0.12). Thus, we did not run Hardik's seeds again.
- Our solve times are approximately 1.6 times longer than Hardik's times on his Mac. This does not
  change the ratios. But more of our solves reach the 900 s limit (68, Hardik: 17).

Plots: `runs/medium_official/ratio_hist.png` (task 3 histogram), `runs/medium_official/time_scatter.png`,
`runs/comparison/ratio_vs_pdf.png` (our histogram on top of Hardik's histogram).

## Notes

- The solve time changes with the CPU. Compare times only from the same machine.
- One instance can have a very different time on a different CPU, because SCIP can take a
  different search path. Thus, compare the distribution of many instances, not single instances.
- The official tester stops a solve after 3600 s. Hardik stopped a solve after 900 s.
  `compare.py` cuts our times at 900 s before it compares them with Hardik's results.
