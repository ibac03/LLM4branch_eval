# LLM4Branch evaluation

Experiments on [LLM4Branch](https://github.com/hzn18/LLM4Branch)'s evolved branching policy for
Independent Set on Barabási–Albert graphs (Ecole `IndependentSetGenerator`, affinity 4), following the
future-work items of Hardik's *LLM4Branch Within-Family Variation* report:

1. Run the official Indset policy and SCIP's default RPB (`relpscost`) on the official test instances.
2. Plot the time-ratio histogram (policy / RPB) and study the slowest 10% of instances.
3. Evolve a specialist policy for the hard instances and pick between the two policies online
   from a cheap 50-node probe (task 4, in progress).

## Layout

| Path | What it is |
|---|---|
| `setup/install.sh` | One-time setup on Ubuntu 24.04: Miniforge, the `ecole-fork` conda env (exact versions in `conda-linux-64.lock` + `pip-requirements.txt`), clones the two official repos into `external/` at the commits used and builds the Ecole fork |
| `external/` | *(not in git)* `LLM4Branch` and `LLM4Branch-ecole`, unmodified |
| `within-family-variation/` | Hardik's code exactly as received (first commit), plus our task-4 changes (see below) |
| `scripts/` | Our scripts (table below) |
| `results/tasks1-3_easy/` | Official tester run on all 1300 easy instances (EC2 c7a.8xlarge, 32 cores) and its figures |
| `results/medium_partial/` | 74 medium (n=1000) instances, stopped early; biased toward fast instances |
| `results/task4_pilot/` | 50-node probe on the 1300 easy instances, the hard-set split, and two short evolution attempts |

### Scripts

| Script | Purpose |
|---|---|
| `run_tester.py` | Runs the official `tester_parallel` (RPB + policy) → `results/runs/<level>.log` |
| `analyze_repo_log.py` | Tester log → per-instance CSV, summary JSON, ratio histogram, time scatter |
| `probe_selector_features.py` | Runs the policy for the first 1/10/25/50 nodes of each instance and records the gap and time |
| `make_specialist_split.py` | Slowest 10% by policy time → 4 train / 4 valid / 122 held-out test instances |
| `selector_signal.py` | How well the probe features predict the hard set |
| `run_specialist_evolution.sh` | Launches the specialist evolution (needs an API key, see below) |
| `make_tuned_program.py` | Bakes the tuned PARAMS of an evolved program into the file the tester reads |
| `selector_sim.py` | Simulates RPB / original / specialist / selector / oracle on the 1300 instances |

## Setup

```bash
git clone https://github.com/ibac03/LLM4branch_eval.git && cd LLM4branch_eval
bash setup/install.sh            # ~10-20 min, most of it compiling Ecole
conda activate ecole-fork
```

## Reproduce tasks 1-3

```bash
N=1300 python scripts/run_tester.py easy
python scripts/analyze_repo_log.py results/runs/easy.log results/runs/easy
```

Solve times depend on the CPU, so compare timings only within one machine. Node counts are
deterministic and should match `results/tasks1-3_easy/easy.log` exactly.

Results on EC2 (1300 easy instances, n=750): policy mean 10.67 s vs RPB 24.50 s (geometric means
7.04 vs 17.61 s); median time ratio 0.378, 90th-percentile ratio 0.594; 14 instances where the policy
is slower than RPB. The slowest 10% of instances account for 44.8% of the policy's total time.

## Task 4 (in progress)

Changes to Hardik's code (second commit, `git diff HEAD~1 -- within-family-variation` shows them):

- `evaluator_specialist.py`: our train/valid seeds; plain `fork` multiprocessing on Linux
  (the forkserver failed inside openevolve's worker processes with `ChildProcessError`).
- `openevolve/llm/openai.py`: headers required by the OpenCode Go API.
- `config_specialist_easy.yaml`, `run_specialist_easy.py`: easy-instance version of his specialist run
  (DeepSeek v4.1 flash via OpenCode Go, 40 iterations, all cores).
- `configs/my_key_set.example.yaml`: copy it to `my_key_set.yaml` (git-ignored) and add the key.

Pilot status: the 50-node probe works (gap after 50 nodes has Spearman 0.912 with policy time and
flags 74 of the 130 slowest instances). Re-tuning the original policy's parameters on the hard set gave
PARAMS [0.0, 0.0, 1.5058, 2.0] (a candidate "cheap specialist"). No LLM-written candidate passed
the stage-1 screen yet.

Next steps:

1. Cap the stage-1 solves at ~210 s; bad candidates otherwise run for the full hour.
2. Rerun the evolution, then test the re-tuned original and the best specialist on the 122 held-out
   hard instances with the official tester.
3. Run `selector_sim.py` to see whether probe-then-switch lowers the average solve time.
