# Backlog

## -1. Is a different family more prone to within-family structural variation than SetCover?

Untested hypothesis, not evidence -- we've only deeply investigated
SetCover this session (categories A-J). Best theoretical candidate:
**Independent Set**, whose standard generator
(`ecole.instance.IndependentSetGenerator(graph_type=barabasi_albert)`) is
a *scale-free graph generator* -- heavy-tailed node degree is already
baked into the standard/baseline population, unlike SetCover's
homogeneous i.i.d. generator. That could mean IndSet's "standard" tier
already spans a wider structural range, making sub-populations more
natural to find (vs. SetCover, where we had to construct heavy-tailed
degree artificially as Category A, and it didn't even work). Weaker
secondary signal: IndSet's released policy uses only 4 features (smallest
of the 4 families), Facilities uses 10 (most) -- unclear what this implies
either way. Would require repeating the whole categorical investigation
for a new family from scratch; not started.


## 0. Cross-family heterogeneous stream (cheap complementary data point — NOT primary evidence, see caveat)

**Caveat, established via discussion — read before treating this as
strong evidence:** in most real deployments, which *family* an instance
belongs to is usually free/trivial to know from context (which
application/pipeline generated it), even without inspecting the model
itself. If that's true, family-level routing needs no online learning at
all -- you'd just look up the right dedicated policy. This experiment is
only strongly motivated in settings with genuinely generic/shared solving
infrastructure (submit-a-.mps/.lp-file services) where no such label
exists.

**Why it's still worth doing, but as a secondary result:** even where a
family label IS available, it only tells you the nominal family, not
whether THIS instance's structure matches what that family's policy was
tuned for -- which is exactly the Category-G finding (a correctly-labeled
SetCover instance can still be clustered-and-multicovered underneath, and
the label doesn't help). That within-family argument has no equivalent
escape hatch (there's no free "structural sub-type" label the way there's
sometimes a free family label) and should stay the primary pillar of the
evidence. Treat this cross-family experiment as a complementary,
lower-stakes data point, not the sharper version of the argument.

**Idea:** instead of testing heterogeneity *within* SetCover (Category G vs
standard), test it *across* the four families LLM4Branch already ships
dedicated policies for: setcover, cauctions, indset, facilities. The real
question isn't "does a SetCover-tuned policy work on CA" (expected-bad,
already anticipated by the paper's own per-family design) — it's **what
happens on a genuinely mixed stream where the solver doesn't know in
advance which family the next instance belongs to**, i.e. no upstream
label to route by. That's a realistic deployment scenario the paper's
per-family design doesn't address, and it's the same
portfolio/online-selection problem as Category G, just at coarser
granularity.

**Prediction:** the oracle gap here should be much larger than anything
found within SetCover (3-8%), since cross-family feature-semantic
mismatch should be starker than within-family structural mismatch.

**Confirmed feasible (checked the 4 program.py files):** each domain's
released policy is a genuinely different evolved program, not just
different weights on one shared formula -- setcover uses 6 features,
cauctions 7, indset 4 (with `BOUNDS=[0,2]` instead of `[0,1]`), facilities
10. `load_program()` reads whichever `program.py` is pointed at
dynamically, so cross-applying any policy to any family's instances works
with zero code changes -- this is comparing structurally distinct evolved
programs across domains, not just a weight-transfer test.

**Experiment (no new generator/design work needed — everything already
exists):**
- 5 arms: RPB (`relpscost`) + the 4 released dedicated policies
  (`program/setcover/program.py`, `program/cauctions/program.py`,
  `program/indset/program.py`, `program/facilities/program.py`)
- 4 families × medium tier, 80 instances each (matching the paper's own
  convention, via `ecole.instance.*Generator` / `create_instance()` /
  `tester_parallel.py --dataset {setcover,cauctions,indset,facilities}
  --medium`)
- Cross-apply every arm to every family's 80 instances (20 combinations,
  ~1600 solves total) -- reuse `tester_parallel.py`'s existing
  `--program_path`/`--dataset`/`--medium` support directly, one
  arm×family combination at a time, fully sequential (same memory-safety
  + fair-timing discipline as the Category-G validation).
- Compute: geomean per arm on each family, the 5-arm oracle on each
  family alone, and the 5-arm oracle on the full 320-instance mixed
  stream vs. the best single fixed policy used universally.

## 1. Find a category where re-tuning makes the policy *worse* on standard SetCover

**Goal:** so far, every re-tuned Category-G policy (seeds 123, 1, 7, 42) still
beat the original released policy on standard SetCover medium — just by
different margins, and with a genuine rank-reversal between seeds (Tuned-7
best on standard/worst-of-the-tuned-set on Category G, Tuned-1 the reverse).
That's real evidence for the portfolio case, but it would be much stronger
evidence if we found a category where re-tuning *for that category*
actively **regresses** standard-SetCover performance below the original —
not just "less good than sibling retunes," but a genuine net loss.

**Why this would matter:** it would mean specialization has a real,
unavoidable cost (not just a different optimum), making "just retune
periodically" clearly worse than maintaining a portfolio.

**Mechanism hypothesis:** `params[0]`/`params[1]` (fractional-value
"centrality" — i.e. "branch on the most fractional variable") are likely
the single most load-bearing, universally useful signal for standard
SetCover. If a category makes that signal completely non-discriminating
(every live candidate tied on fractional value throughout much of the
search), BO re-tuning has no reward signal keeping that weight high, and if
it drifts down, standard-SetCover performance should suffer as a
consequence.

**Attempt #1 (failed, documented so we don't repeat it):** built
`category_i_oddcycle_generator.py` — vertex cover on a disjoint union of
small odd cycles (rows=edges, columns=vertices), reasoning that the LP
relaxation's symmetric "x=0.5 for every vertex" solution would make every
candidate maximally and identically fractional. **This was wrong.** The
"all-0.5" point is a convex combination of the polytope's actual vertices,
not a vertex itself — simplex returns an actual vertex, which for an odd
cycle has only *one* fractional variable per cycle (the rest snap to clean
0/1 alternating around the cycle). Confirmed via a lightweight root-node
check: `action_set` came back `None`/`done=True` immediately — the
instance solved at the root with zero branching, the opposite of the
intended effect. Caught cheaply (one root-node observation, no real solve)
before wasting a real batch on it.

**Next attempt — not yet designed.** Need a mechanism that actually
survives into the branching tree (not just the root LP), where local
fractional-value/centrality genuinely carries no information for many
live candidates at once, for many nodes, without SCIP just presolving or
rounding its way past it.

## 2. The "300 fresh seed" validation protocol — reusable, run again once #1 produces a category

Once a candidate category from #1 exists (with an easy tier + at least one
re-tuned policy), re-run the exact same validation methodology already
built for Category G:

1. Pick 150 **fresh, non-overlapping** seeds for standard SetCover medium
   (continue from whatever the last-used standard SetCover seed range
   ended at) and 150 fresh seeds for the new category's medium tier.
2. Evaluate all arms (RPB, Original, and every re-tuned variant so far) on
   both seed sets, **fully sequentially** (one arm at a time, no
   parallelism across arms/domains) — this was an explicit choice for
   fair, contention-free `proc_time` measurement, not a shortcut.
3. Compute per-arm geomean time, win counts, and the multi-arm oracle gap
   on: standard-only, new-category-only, and the combined
   (standard + new-category) heterogeneous stream.
4. Compare the mixed-stream oracle gap against the isolated-domain gaps —
   the heterogeneous stream should show the largest gap if the arms are
   genuinely differentiated (this is what happened with Category G: 3.2%
   standard-only, 5.1% Category-G-only, 7.5% mixed).

Reusable scripts already built for this (generalize the arm/category
names when applying to a new category):
- `examples/brancher/experiments/structural_generalization/eval_arm_standard_fresh.py`
- `examples/brancher/experiments/structural_generalization/eval_arm_categoryg_fresh.py`
  (currently Category-G-specific — would need a small generalization to
  point at a different generator module)
- `/Users/hs/Desktop/scip-build/run_fresh_validation.sh` (the 12-step
  sequential driver)
- `/Users/hs/Desktop/scip-build/compute_full_multiarm.py` (the analysis
  script — currently reads the original 80+80 logs, needs pointing at
  the fresh CSVs instead)

**Status as of writing:** the Category-G version of this validation
(150 standard + 150 Category-G, 6 arms: RPB/Original/Tuned-123/1/7/42) is
currently running (`run_fresh_validation.sh`, fully sequential by design,
~13-14hr estimated total). This backlog item is about applying the *same
protocol* to whatever new category comes out of item #1, once it exists.
