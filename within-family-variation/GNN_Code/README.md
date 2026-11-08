# GNN branching baseline — reproduction package

This is a standalone package (no dependency on the rest of the LLM4Branch
repo) to reproduce the GNN branching-policy baseline used as the
mixed-model-training ablation against the LLM4Branch/Evolved-* policy
portfolio (see `portfolio.md` in the main repo, section 6). It trains
**three separate GNNs** — one specialized on each of three structural
regimes (BA, RR, WS) — via imitation learning against strong branching,
then cross-tests all three on all three domains' held-out test sets.

The GNN itself is a from-scratch PyTorch port of the bipartite GCNN
architecture from Gasse et al. 2019 ("Exact Combinatorial Optimization
with Graph Convolutional Neural Networks") — see `model.py`.

## Files

| File | Purpose |
|---|---|
| `scip_utils.py` | Standalone SCIP-param / normalization / geomean helpers |
| `instance_generators.py` | BA/RR/WS instance generators (training tier + held-out test tier), exact test seeds |
| `model.py` | `GNNPolicy` — the bipartite GCNN architecture |
| `generate_data.py` | Generate imitation-learning training data (per domain) |
| `train.py` | Train a GNN on one domain's data |
| `select_checkpoint.py` | Pick the best checkpoint by real solve time (not loss) |
| `crosstest.py` | Full 3×3 evaluation: each trained GNN × each test domain |

## Setup

1. Install dependencies — see `requirements.txt`. **Read it fully before
   installing**: `ecole` is not the stock PyPI package, it's a custom
   fork you'll need to build yourself against your own SCIP install. This
   is the single biggest risk to getting this running at all, let alone
   reproducing exact numbers — budget real time for this step.
2. Sanity-check your install:
   ```
   python instance_generators.py
   ```
   Should print instance sizes for all three domains' train/test tiers
   with no errors (expected output: BA train n_vars=400, test n_vars=1000;
   RR train n_vars=200, test n_vars=250; WS train/test both n_vars=200).

## Running the full pipeline (per domain — repeat for ba, rr, ws)

**Always smoke-test at tiny scale first.** This pipeline has genuine
tractability traps (see "Known issues" below) — a small dry run costs
minutes and catches environment/config problems before you commit to a
run that could otherwise take many hours before you find out something's
wrong:
```
python generate_data.py --domain ba --split train --num_samples 100 --seed_start 0
```
Confirm it finishes in a couple minutes with a `[train] done: 100 samples
from N instances` line, not a multi-minute hang.

**1. Generate training data** (repeat for `--domain ba`, `rr`, `ws`):
```
python generate_data.py --domain ba --split train --num_samples 100000 --seed_start 0         --num_workers 6
python generate_data.py --domain ba --split val   --num_samples 20000  --seed_start 100000000 --num_workers 6
```
On the original machine (6 workers, Apple Silicon, BA domain — the
biggest/slowest of the three), this took **~10h15m total** (train+val).
RR and WS use smaller training-tier instances (`n_nodes=200` vs BA's 400)
so should be meaningfully faster, though this wasn't directly measured —
budget real time regardless and expect it to run for hours, not minutes.

**2. Train** (repeat per domain):
```
python train.py --domain ba --epochs 50 --batch_size 16 --device cpu
```
**Use `--device cpu`.** On the original machine, `--device mps` (Apple
Silicon GPU) went 8.5+ hours without completing a single epoch and had to
be killed; CPU training was dramatically faster and stable (~6.6-8
min/epoch, ~5.5-6.5h total for 50 epochs on BA). If you're on CUDA
hardware, `--device cuda` is untested by us but worth trying — just
don't assume any particular accelerator is faster without checking, and
watch the per-batch log lines (printed every 200 batches) for a stable
rate before committing to the full run.

Checkpoints save every epoch to `checkpoints_{domain}/epoch_NNN.pt`.

**3. Select the best checkpoint** (by real solve time, not loss — repeat per domain):
```
python select_checkpoint.py --domain ba --max_workers 6 --num_instances 20
```

**4. Cross-test all three trained GNNs on all three domains:**
```
python crosstest.py \
    --ba_checkpoint checkpoints_ba/epoch_XXX.pt \
    --rr_checkpoint checkpoints_rr/epoch_XXX.pt \
    --ws_checkpoint checkpoints_ws/epoch_XXX.pt \
    --max_workers 6
```
(use the checkpoint paths `select_checkpoint.py` reported as time-optimal
for each domain). This prints the full 3×3 timing matrix, each GNN's
pooled 90-instance geomean, an oracle row (per-instance min across the
three GNNs), and win counts — directly comparable to the
Original/Evolved-RR/Evolved-WS rows in the main repo's oracle table.
**Note:** BA's test tier uses a 3600s-per-instance cap (matching the
paper's own convention); RR/WS use 120s. A poorly-performing GNN on BA
can genuinely take a long time per instance — this is expected, not a
bug (see Known issues).

## Known issues (read before you hit these yourself)

**1. Data generation at the wrong instance size is catastrophically
slow.** A first attempt at BA using ecole's own `indset/easy` tier
(n_nodes=750, affinity=4) with no time cap took 2.5+ minutes without even
finishing 20 samples — strong branching (the data-gen "expert") evaluates
every fractional candidate's LP relaxation per node, far more expensive
than normal branching. Extrapolated, that would have taken 150-250+ hours
for the real 100k+20k target. Fixed by (a) using each domain's own
smaller, calibrated training tier (see `instance_generators.py`'s
docstring) and (b) capping every data-gen episode at 180s
(`generate_data.py`'s `TRAIN_TIME_LIMIT`). Don't loosen either of these
without expecting a similar blowup.

**2. `tester_parallel.py`-style silent failures aren't a risk here**
(this package's `evaluate_one`/`solver` logic always propagates real
exceptions), but if you ever port pieces back into the main repo's
`tester_parallel.py`, note that its `solver()` swallows exceptions
without setting `validity=False` — a broken policy can silently report a
fake 0.0s "success" instead of erroring. Not applicable to the scripts in
this folder, just a heads-up if you go cross-referencing.

**3. MPS (Apple Silicon GPU) training can stall for hours with zero
progress, and — critically — the *original* version of `train.py` had no
per-batch logging, so this was invisible until we added it.** A training
run on `--device mps` went 8.5+ hours without completing epoch 1 and had
to be killed blind. The per-batch progress lines (every 200 batches, with
running loss/accuracy/rate/ETA) now in `train.py` exist specifically
because of this — if you ever strip that logging out "for cleanliness,"
you're removing the only way to tell "almost done" from "badly stuck."
Root cause wasn't fully isolated (likely MPS memory growth/fragmentation
on a long-running job, compounded by system-wide memory pressure from
other running applications) — `--device cpu` was simply faster and
stable on the original hardware, so that's the safe default here.

**4. Training is not deterministic without `--seed`.** The original
`train.py` didn't set a torch seed at all — this version does
(`--seed`, default 0), but a *different* seed, PyTorch version, or
hardware backend will still produce a different trained model even with
identical data. "Reproduce our exact GNN" really means "reproduce our
pipeline and get a GNN with the same properties," not bit-identical
weights, unless your full environment (Python, torch, ecole/SCIP build,
OS/hardware) matches ours exactly.

**5. `crosstest.py`'s BA cells can be slow, especially for an
undertrained/early checkpoint.** BA's held-out test tier
(`n_nodes=1000, affinity=4`) is meaningfully harder than RR/WS's, and a
GNN that hasn't converged well can produce pathological branching that
eats the full 3600s cap per instance. This mirrors what was observed with
the fixed-policy portfolio too (e.g. a BA-retuned Evolved-WS policy took
~5h47m for 30 BA instances in the original run) — it's a property of the
BA test tier being hard for a bad policy, not a bug in this code.

## Expected results shape (for sanity-checking your reproduction)

You should see roughly:
- **Within-domain performance should beat cross-domain** for the same
  trained-on-X-tested-on-X cells vs. off-diagonal cells, on average — but
  don't expect a clean diagonal-dominant matrix; the fixed-policy
  portfolio in the main repo showed real asymmetric transfer (e.g. an
  RR-specialized fixed policy transferred badly to WS, but a
  WS-specialized one transferred *well* to RR, despite RR/WS sharing a
  similar degree distribution) — the GNN baseline may or may not show the
  same pattern, and that comparison (does a more expressive model avoid
  this asymmetry, or reproduce it?) is exactly the point of this
  ablation.
- BA is the hardest domain across the board (largest instances, highest
  time cap) for any policy, evolved or learned.
