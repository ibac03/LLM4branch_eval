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
| `scripts/` | Tasks 1 to 3: `run_tester.py`, `analyze.py`, `compare.py`. Task 4: `run_portfolio.py`, `probe_features.py`, `task4_analysis.py`, `task4_racing.py`, `tune_params.py`. |
| `policies/` | One file for each branching policy that task 4 compares. |
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

## Results (task 4): hard instances, routing, cascade, new policies

All runs: Carolina Cloud VM, one solve for each core. "Original" is the LLM4Branch Indset policy.
Easy = BA graphs with 750 nodes (1292 test instances, 300 s limit). Medium = 1000 nodes (the first 400
test instances, 900 s limit). A solve that does not finish counts as the limit.

### 1. We can find the hard instances early

A model gets cheap features and predicts the solve time (medium, 1293 instances from tasks 1 to 3).

| Features | Slowest 10% found (of 129) | Cost (% of the mean solve time) |
|---|---|---|
| Graph only (before the solve) | 45 | 0 |
| + root node | 72 | 4.3 |
| + first 25 nodes | 89 | 6.1 |
| + first 50 nodes | 91 | 6.7 |

The best single feature is the gap after 50 nodes (Spearman 0.84 with the solve time). Graph
features are weak. Plot: `runs/task4_medium/hard_features.png`.

### 2. In the BA family, routing between policies does not help

- The same policy with a different random seed changes the node count of an instance by a factor
  of 1.4 (median) and 2.4 (90% quantile).
- Easy: the best of 3 random seeds of the original is 17% faster than one run. The best of the 4 best
  policies is 25% faster. Thus, most of the "which policy wins" effect is random.
- The router selects the fastest policy on approximately 30% of the instances (4 policies, chance is 25%).
  It is not faster than the best single policy (`runs/task4_easy/router.png`, `runs/task4_medium400/router.png`).
- 33 of the 40 slowest instances stay in the slowest 10% with a different seed. Thus, "hard" is a
  property of the instance. But the winner between two good policies is not.

### 3. Between graph families, routing helps

Mix of 200 BA, 200 RR (random regular) and 200 WS (Watts-Strogatz) instances, 120 s limit
(`runs/task4_mixed/`). Mean time in seconds:

| Family | Original | Original, tuned again | Degree | RR policy | WS policy | RPB |
|---|---|---|---|---|---|---|
| BA | 9.4 | 8.3 | 8.7 | 80.5 | 62.6 | 23.0 |
| RR | 20.5 | 21.0 | 24.6 | 4.8 | 8.6 | 4.4 |
| WS | 28.2 | 28.8 | 35.3 | 62.3 | 14.6 | 12.3 |

| System | Mean time |
|---|---|
| Original policy | 19.3 s |
| Best single policy (RPB) | 13.2 s |
| Router with graph features only (no cost) | 8.8 s |
| Oracle (best policy for each instance) | 7.7 s |

The router is 33% faster than the best single policy and gets 79% of the oracle gain.

### 4. Cascade (Hardik's idea): small gain

Start several runs (policy + seed) for 50 nodes, then continue the run that a model predicts to be the
fastest. The probe time of the stopped runs is included.

- The probe selects the faster of two runs of the same instance in 62 to 66% of the cases (50% is chance).
- Easy: one probe costs 33% of the mean solve time. The cascade is slower than one run.
- Medium: one probe costs 10%. The best cascade (race 3 runs, only on the 20% predicted-hard
  instances) is 4% faster than the original. This is less than the gain from a better single policy.

### 5. Better single policies

| Policy | Easy, mean time | Medium, mean time |
|---|---|---|
| RPB | 27.7 s | 147.4 s |
| Original | 12.0 s | 70.6 s |
| Original, other seeds | 11.9 s, 11.9 s | 68.5 s, 73.5 s |
| Original, tuned again on 4 hard instances (`orig_retuned`) | 11.0 s (-8%) | 57.9 s (-18%) |
| Degree only: one feature, the number of constraints of the variable (`degree`) | 11.2 s (-7%) | 58.5 s (-17%) |
| Original, tuned on 48 instances (`orig_tuned48`) | 10.4 s (-14%) | 60.8 s (-14%) |
| New policy from the DeepSeek evolution (`evolved`) | 10.8 s (-10%) | 60.9 s (-14%) |

- All four new policies are faster than the original on easy and on medium. The gain is larger than
  the seed noise.
- DeepSeek evolution: 41 candidates in 95 minutes. One candidate passed the two tests. The other
  candidates were 2 to 10 times slower than the original on the hard training instances.
- A policy with one feature (`degree`) is as fast as the evolved 4-feature policy on BA graphs.

### How to run task 4 again

```bash
python scripts/run_portfolio.py --level easy --seeds 999999999:1300 --cap 300 --out runs/task4_easy/portfolio.csv \
    --policies orig,orig_retuned,degree,cauctions,rpb,evolved,orig_tuned48 --env-seeds "orig:1,2;orig_retuned:1;degree:1"
python scripts/probe_features.py --level easy --seeds 999999999:1300 --out runs/task4_easy/features.csv
python scripts/task4_analysis.py --dir runs/task4_easy --cap 300      # policy table, features, router
python scripts/task4_racing.py --dir runs/task4_easy --cap 300        # cascade
python scripts/tune_params.py --policy orig --out-policy orig_tuned48 # parameter tuning on 48 instances
```

`policies/` has one file for each policy. The DeepSeek evolution uses Hardik's code
(`within-family-variation/.../run_specialist_easy.py`) and an API key in `configs/my_key_set.yaml` (not in git).
