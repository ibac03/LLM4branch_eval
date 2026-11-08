# Portfolio: LLM-Evolved Branching Specialists Across Structural Regimes

Status as of 2026-09-02. This document tracks the paper's central thesis, methodology,
and results so far. It is a living document — update it as new rows/domains are added.

## 1. The paper idea

**Thesis:** A single fixed MILP branching policy (LLM4Branch's per-family released
`program.py`, or any one fixed reweighting of it) is *structurally insufficient* — real
instance populations vary in ways that shift which branching signal is informative, and
no single policy dominates across that variation. The fix is not "tune harder" but a
**portfolio of genuinely different policy structures** (different feature sets, not just
different weights on the same features), selected per-instance or per-regime.

**Mechanism:** use LLM-driven program synthesis (openevolve + DeepSeek-R1, two-stage
cascade: cheap stage-1 filter on a few seeds, then embedded Bayesian-optimization
parameter tuning + stage-2 validation) to *discover* new candidate structures for a given
structural regime, rather than hand-designing them. This is both the experimental tool
(finding genuine specialists) and part of the pitch (LLMs as a practical way to search
policy-structure space cheaply, vs. hand-engineering a new evolved program per regime).

**Evidence strategy — the "oracle gap":** for a combined test set spanning several
structural regimes, compute the geomean solve time of (a) each fixed policy used
universally, and (b) the *oracle* that always picks whichever fixed policy is best for
each instance. A large, persistent gap between the best single fixed policy and the
oracle is the headline empirical result: proof that no fixed policy is enough, and a
quantified upper bound on what a portfolio/online-selector could recover.

### Why LLM-driven evolution specifically (not just BO retuning)

Early experiments (documented below) showed that cheap BO retuning of an *existing*
policy's fixed feature structure onto a new regime (RR) produced a degenerate result —
BO could only reweight, and reweighting the existing 4-feature structure had *no*
exploitable signal on RR at all (near-50/50 coin flip). Only genuinely new features,
discovered via LLM-driven structural search, found real signal. This is the load-bearing
distinction in the paper: "Tuned-via-RR" (cheap BO reweighting of Original's structure,
found degenerate) vs. "Evolved-RR" (genuine LLM-discovered new structure) are not
duplicates — they demonstrate that specialization requires structural search, not just
parameter search.

## 2. Structural regimes studied

Three MIS structural regimes so far, chosen to isolate different sources of structural
variation:

| Regime | Generator | What it isolates |
|---|---|---|
| **BA** (Barabási–Albert) | `ecole.instance.IndependentSetGenerator(graph_type=barabasi_albert)` | Heavy-tailed/hub-heavy degree distribution — LLM4Branch's own released "standard" tier. Original's `constraint_count` (=degree) feature is highly informative here. |
| **RR** (random-regular) | custom `networkx.random_regular_graph` generator (`category_rr_indset.py`), n=250, degree=3 | Every vertex has *exactly* the same degree — removes degree heterogeneity entirely, the sharpest possible test of whether Original's degree-reliant signal is BA-specific. |
| **WS** (Watts–Strogatz, clustered/small-world) | custom `networkx.watts_strogatz_graph` generator (`category_ws_indset.py`), n=200, k=20, p=0.25 | Uniform-ish degree (like RR) but *strong local clustering* (many triangles/overlapping neighborhoods) — isolates whether specialization is driven by raw degree distribution (RR) vs. higher-order topology/clustering (WS), holding degree roughly fixed. |

Planned: scale to 6–7 structural variants total (methodology below is designed to make
this sustainable rather than O(N²)).

## 3. Methodology (established rules)

These were worked out iteratively and are now fixed conventions for all future rows/domains:

**Evolution always trains on an easier tier than the test tier.** Never train an
openevolve run on the same instance parameters as the held-out test set — that makes
evolution/BO prohibitively slow (each stage-2 BO call solves `train_num` instances per
candidate, over many candidates/iterations). But the training tier also can't be *too*
easy, or there's no gradient to discriminate policies on (a WS attempt at p=0.15 gave a
mean node count of only ~600, one seed as low as 43 — rejected as too shallow/noisy).
RR trains at `n=200,degree=3` (geomean≈2.58s) vs. tests at `n=250,degree=3`
(geomean≈29s), an ~11x ratio. WS trains at `n=200,k=20,p=0.2` (geomean≈5.09s,
mean_nodes≈3798) vs. tests at `n=200,k=20,p=0.25` (geomean≈30s, mean_nodes≈37189), an
~6x ratio — both chosen to keep a real, non-trivial search tree while staying cheap.

**Oracle table: rows are structures, not fixed parameter vectors.** A row (e.g.
"Evolved-RR") is a feature-set/logic, and its geomean over the combined instance set can
legitimately be made of *different parameterizations* of that same structure on
different domains — native evolved params on one domain, a cheap BO retune on another.

**Per-domain cell: native params first, retune only on a real failure.** For each
(row, domain) cell, try the structure's own native/already-evolved parameters first.
Only retune (cheap BO) if native params are screened as clearly failing — compared
against RPB (random-priority-branching) on a small sample (e.g. 4 seeds),
`speedup = RPB_geomean / policy_geomean`. The retune trigger is a real failure (e.g. "the
full test set would take ~10 hours"), not merely `speedup < 1`. This applies uniformly,
including to the Original row. A retune folds into the *same* row, not a new one.

**Same seeds across all rows, per domain.** "30 BA instances" means one fixed set of 30
seeds for that domain, and *every* row is evaluated on that exact same set — this is what
makes the Oracle row's per-instance `min()` valid and makes cross-row comparisons paired.

**Final geomean: pool raw per-instance times, one `geometric_mean_stable()` call.**
Never shortcut via geomean-of-domain-geomeans — `geometric_mean_stable()` uses a
*shifted* formula (`exp(mean(log(x+1)))-1`), which breaks the clean multiplicative
decomposition a geomean-of-geomeans would need. Pool all raw per-instance times across
domains for a row into one flat list (equal group sizes per domain), then call
`geometric_mean_stable()` once directly.

**Oracle row:** per-instance `min()` across all fixed-policy arms' raw times on that
instance, then pooled and geomean'd the same way as any other row.

## 4. Results so far

### 4.1 Evolved-RR: genuine structural specialization found

- **Original** (BA's released policy, 4 features: `obj_coeff`, `reduced_cost`,
  `fractional_part`, `constraint_count`) has *no* exploitable preference on RR (near
  50/50 win rate across parameter extremes) — confirmed via a 50-seed paired test.
  `obj_coeff`/`reduced_cost` were separately confirmed dead features for unweighted MIS
  across BA/RR/ER generally (15-config, 30-node-deep verification).
- **openevolve + DeepSeek-R1**, run on RR (50 iterations, two-stage cascade evaluator),
  discovered a genuinely new 5-feature structure (`fractional_dist`, `pseudocost_sum`,
  `pseudocost_product`, `dynamic_ratio`, `dynamic_mean` — features 37/42/43/48/51):

  ```
  score = params[0]*pseudocost_product*dynamic_ratio + params[1]*fractional_dist
        + params[2]*pseudocost_sum + params[3]*dynamic_mean
  ```

- **Cross-domain check (paper's own seed convention, seed=999999999, 30 instances,
  1-hour SCIP time limit):**
  - Original on BA: geomean **25.49s** (clean, mean_gap=0)
  - Evolved-RR (BA-tuned via BO on Original's structure — "Tuned-via-RR"): degenerate,
    confirmed as expected given the dead-feature finding above.
  - Evolved-RR (native RR params) on BA: **~22x worse** than Original even after a
    full BA-specific BO retune of the *same* 5-feature structure (geomean **569.26s**,
    mean_gap=0.00254) — specialization is not fixable by retuning, it's structural.
  - Evolved-RR on RR (n=250, seeds 1000–1029): geomean **4.56s**, strong.

This is the core specialization result: Evolved-RR is genuinely good on RR and
genuinely bad on BA regardless of retuning effort, and Original is the reverse.

### 4.2 Combined BA+RR oracle gap (60 instances)

- Oracle-gap headline: **1.91x** (oracle geomean vs. best single fixed policy's geomean),
  computed on pooled raw per-instance times across the 60 combined BA+RR instances
  (30 each), near-even win split (31/29) between Original and Evolved-RR across
  instances — i.e. neither policy dominates, consistent with the thesis.

### 4.3 WS calibration

Watts–Strogatz MIS is theoretically *easier* than RR at low rewiring (clustering
collapses to near-trivial "pick ~1 per near-clique" decisions) — confirmed empirically:
trivially easy (mean_nodes~1) at p<0.1 regardless of k up to k=32; only genuinely hard
at higher k *and* higher p. Final calibrated test tier: **n=200, k=20, p=0.25**
(geomean=30.08s, mean_nodes=37189, all-optimal for Original; geomean=20.32s,
mean_nodes=21830 for Evolved-RR-tuned-on-BA) — p=0.3 already shows a timeout, so p=0.25
is the ceiling of tractability.

### 4.4 WS cheap screen (native params, 4 calibration seeds)

| Arm | geomean | speedup vs RPB |
|---|---|---|
| RPB (reference) | 16.00s | 1.00x |
| Original (native) | 29.99s | 0.53x |
| Evolved-RR (native) | 51.30s | 0.31x |

Neither existing structure beats RPB on WS, but neither is anywhere near the ~10-hour
retune-trigger threshold — so **both keep native params** for the WS column (no retune
needed). This screen result is itself evidence that WS is a genuinely fresh structural
challenge, motivating a dedicated Evolved-WS structure.

### 4.5 Evolved-WS: discovery, screening, and retuning

openevolve + DeepSeek-R1 run on WS (50 iterations, training tier `n=200,k=20,p=0.2` —
see the training-tier rule above) discovered an 8-feature structure (features
22/37/38/43/52/54/55/85 — constraint participation, fractional distance, ceiling
distance, pseudocost product, dynamic/static mean & max ratios, min coefficient/RHS
ratio, dual-weighted active constraints):

```
cluster_dynamic    = params[0] * (f52 * f54) * sqrt(f22 + 1e-6)
fractional_impact  = params[1] * (f37 * (1 - f38))**1.5
pseudocost_synergy = params[2] * f43 * (f52 + f54)
dual_constraint    = params[3] * f85 * f55
base_pseudocost    = params[4] * f43
log_dynamic        = params[5] * log1p(f22) * f54
score = cluster_dynamic + fractional_impact + pseudocost_synergy
      + dual_constraint + base_pseudocost + log_dynamic
```

The best-found parameterization zeroed out 3 of the 6 terms (`cluster_dynamic`,
`dual_constraint`, `log_dynamic`), leaving `fractional_impact`, `pseudocost_synergy`,
and `base_pseudocost` as the active signal.

**Cheap screen (4 seeds each, vs. RPB), deciding native-vs-retune per domain:**

| Domain | Evolved-WS (native) | RPB | Speedup | Decision |
|---|---|---|---|---|
| RR | geomean=3.96s, 4/4 optimal | geomean=3.76s | 0.95x | **Keep native** |
| BA | geomean=123.17s, **0/4 optimal** (all timelimit) | geomean=97.70s, 2/4 optimal | 0.79x | **Retune** — the 0/4-optimal pattern (worse than RPB's own 2/4) is the clear "failing badly" signal the methodology calls for |

BA-specific BO retune (same recipe as Evolved-RR's BA retune: `n=400,affinity=5`
training tier, 50 calls) converged to
`theta=[2.0, 0.0, 3.0, 0.6526, 1.7536, 2.0]` (best training mean_nodes=5496.5).

**A real bug, caught and fixed along the way:** the saved
`program_evolved_ws_ba_tuned.py` was initially missing `import numpy as np` — the
`write_program()` helper (inherited from the RR precedent, where it was harmless
because that structure's body used only `+`/`*`) strips everything before
`def score_function`, and Evolved-WS's body calls `np.sqrt`/`np.log1p` directly. Worse,
this failed **silently**: `tester_parallel.py`'s `solver()` catches all exceptions
internally but never sets `validity=False`, so every solve logged a fake "success" at
`Proc Time: [0.0]` instead of erroring. Caught via a sanity check on the eval log, fixed
in both the saved program file and `write_program()`, and reran. General lesson for any
future `write_program()`-style saved-policy file: always explicitly re-add
`import numpy as np` regardless of whether the current structure appears to need it, and
never trust a zero-time "success" from this codebase's `evaluate()`/`solver()` without
checking the actual per-instance node counts look real — the exception-swallowing bug is
in `tester_parallel.py` itself, not fixed at the source.

### 4.6 WS held-out test-set result (30 fresh seeds, 200–229; all native params)

| Policy | Geomean | Optimal rate |
|---|---|---|
| Original | 25.38s | 30/30 |
| Evolved-RR | 40.29s | 29/30 (seed 213 hit timelimit) |
| **Evolved-WS** | **11.73s** | 30/30 |

Evolved-WS wins **29/30** WS instances outright (Original wins the other 1; Evolved-RR
wins none) — a clean, unambiguous specialization result on its home domain.

### 4.7 Final 4-row oracle table (90 combined BA+RR+WS instances)

Per-domain geomeans feeding into each row (native unless marked *tuned*):

| Row | BA | RR | WS |
|---|---|---|---|
| Original | 25.49s | 16.37s | 25.38s |
| Evolved-RR | 569.26s *(tuned)* | 5.19s | 40.29s |
| Evolved-WS | 1022.22s *(tuned)* | 5.08s | 11.73s |

Pooled over all 90 same-seeded instances (one `geometric_mean_stable()` call per row on
raw pooled times, per the methodology above):

| Row | Pooled geomean | Win count (/90) |
|---|---|---|
| **Original** | **21.98s** ← best single fixed policy | 32 |
| Evolved-WS | 41.95s | **43** ← most wins |
| Evolved-RR | 51.63s | 15 |
| **Oracle** | **11.20s** | — |

**Oracle gap: 1.96x.** Even the best single fixed policy (Original, which happens to be
decent-but-not-great everywhere) leaves ~2x on the table versus an oracle that always
picks the right specialist. No policy dominates: Evolved-WS wins the most individual
instances (43/90) without having the best pooled geomean (its BA cost is severe enough to
drag its geomean up, even though it never "wins" BA instances), Original has the best
average without winning the most instances, and Evolved-RR is worst on both counts but
still wins 15/90 instances nobody else does better on. This three-way non-domination is
itself part of the evidence: the case for a portfolio doesn't rest on one policy being
uniformly bad, it rests on no policy being uniformly good.

## 5. Online learning extension (proposed, not yet built)

Discussed as the natural next step to turn this from a diagnosis into a method:

**Core mechanism — contextual bandit portfolio selector.** Replace a one-shot offline
classifier with an online learner: context = cheap root/presolve meta-features (degree
variance, clustering coefficient, whichever separate BA/RR/WS), arms = the portfolio
(Original, Evolved-RR, Evolved-WS, ...), reward = −normalized solve time, regret =
cumulative gap vs. the oracle over the instance stream (reusing the oracle-gap metric
above as the natural regret baseline). Candidates: LinUCB, Exp4, Thompson sampling.

**Free prototype available right now:** the complete 90×3 payoff matrix above is a
logged, full-information bandit environment — every arm's outcome is already known for
every instance, so a bandit simulation (in any instance order, including a simulated
drifting mixture) can be built and evaluated with zero new solves.

**Make non-stationarity the point.** If the stream is i.i.d. stationary, online learning
barely beats "fit a classifier once" — the real argument for *online* (vs. one-shot
offline) needs a drifting mixture of regimes (e.g. BA-heavy → RR-heavy → WS-heavy →
mixed), showing a bandit tracks the drift while a frozen static selector degrades.

**The distinctive/ambitious version — self-expanding portfolio.** Let persistent bandit
regret concentrated in some context region (a structural regime none of the current arms
handle well — as WS was before `evaluator_ws.py` existed) trigger a new background
openevolve run, adding the resulting specialist as a new arm. This closes the loop
between online selection and the LLM-evolution pipeline already built here, instead of
discarding it once the initial portfolio is "done": *online detection of
policy-inadequacy regions → automatic offline synthesis of new specialists → portfolio
that grows its own coverage over time.*

**Practical caveat:** a real (non-simulated) online system's reward signal costs up to an
hour per pull (full solve time). A live deployment would need a cheap proxy reward (e.g.
node count after a truncated/depth-limited sub-budget) rather than waiting for full
completion — a multi-fidelity/Hyperband-flavored bandit layered on top, left as future
work even if the initial prototype uses full solves via the logged payoff matrix.

## 6. Is this ICLR-worthy? (honest assessment)

**Current state (sections 1–4): workshop tier, not main track.** It's a clean diagnosis —
oracle gap exists, specialization is structural not parametric, LLM evolution finds real
specialists — but it ends at "here's a gap," not "here's a method that closes it," and
main track wants a method.

**What main-track competitiveness would require, roughly in priority order:**

1. **A mixed-model-training baseline ablation — the single highest-priority missing
   piece.** The obvious competing hypothesis: train one more expressive model (MLP/GNN,
   à la Gasse et al.) on the pooled BA+RR+WS data, instead of building a portfolio of
   simple linear specialists. If that single model already closes most of the oracle gap,
   the whole portfolio framing needs rethinking before investing further. This should be
   checked *before* heavy investment in the selector itself.
2. **The selector** (Section 5) — turns diagnosis into a method, a necessary condition
   for main track but not sufficient on its own.
3. **Broader-than-MIS validation.** Everything so far is one problem family (independent
   set) with three synthetic graph generators. At least one more base family (LLM4Branch
   already ships setcover/cauctions/facilities infrastructure — see `backlog.md`'s
   cross-family idea) would blunt the "does this generalize past MIS?" objection.
4. **Explicit positioning against existing learned-branching generalization
   literature.** Learned branching failing to generalize OOD is not itself a novel
   observation (see work following Gasse et al.'s GNN branching, and generalization
   surveys in this space) — the novel part here is the specific mechanism (structural vs.
   parametric failure) and fix (LLM-discovered specialists + oracle-gap framing), not the
   existence of the phenomenon. Needs explicit citation and differentiation, not a
   from-scratch framing.
5. **Sample size / statistics.** 30 instances/domain is fine internally; a submission
   table wants error bars or enough scale that geomean differences aren't noise-level,
   especially given the high per-instance variance already visible in the BA numbers
   above (max_time up to 3811s vs. geomean 1022s for Evolved-WS-tuned-on-BA).

## 7. Backlog / future directions (not started)

See `backlog.md` for additional untested ideas, including:
- Cross-family (setcover/cauctions/indset/facilities) heterogeneous-stream oracle gap,
  as a complementary (not primary) data point — likely a *starker* gap than within-family
  regime switching, but less directly actionable (family is often a free label at
  deployment time, unlike within-family structural sub-type).
- Finding a regime where retuning actively *regresses* standard performance (not just
  "less good than a sibling retune") — would sharpen the case that specialization has an
  unavoidable cost, not just a different optimum.
- Scaling to 6–7 total structural regimes, reusing the cheap-screen-before-retune
  discipline established above to keep this sub-quadratic in cost.
