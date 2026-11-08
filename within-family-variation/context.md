# Within-Family Structural Variation in LLM-Evolved MILP Branching — Project Context

This file has the full context, timeline, code map and reproduction guide for this
research. It was rebuilt on 2026-10-28 from the repo's files, scripts, logs and
result files, because the original working sessions were lost. Work paused around
2026-09-16.

Related documents in this repo:
- `portfolio.md` covers thesis, methodology and results, but only up to 2026-09-02.
  Everything after that date is documented only here.
- `backlog.md` lists untested ideas from the SetCover phase.
- `GNN_Code/` is a standalone package, with its own README, for reproducing the GNN
  baseline.

---

## 0. TL;DR — where things stand

- **Thesis.** No single fixed branching policy is best across the structural
  variation that exists *inside one problem family*. A portfolio of policies with
  genuinely different feature *structures* is needed. LLM-driven program evolution
  is a cheap way to discover those structures.
- **Headline result (Sep 2).** We studied Maximum Independent Set (MIS) on three
  graph regimes: BA (Barabási–Albert), RR (random-regular) and WS (Watts–Strogatz).
  On 90 pooled instances the best single fixed policy has a pooled geomean of
  21.98s. The per-instance oracle reaches 11.20s, an **oracle gap of 1.96×**.
  No policy dominates.
- **After Sep 2:**
  - Tuned SCIP default branching does not specialize.
  - MLP and GBDT imitation baselines reproduce the same off-domain collapse.
  - A difficulty sweep shows BA's apparent robustness is partly an artifact of
    easy test tiers.
  - Re-evolving at BA-matched difficulty failed to beat RPB.
  - A 1,300-instance BA "worst-decile" analysis found that graph structure does
    **not** predict where the BA policy loses. The root/early gap does weakly
    predict it (Spearman ≈ 0.3).
- **Next planned step (set up, never run).** Evolve a *specialist* for BA's
  worst-decile instances: `run_specialist_evolution.py` with `config_specialist.yaml`.
- **Biggest open gaps for a submission:**
  1. A single model trained on *pooled* BA+RR+WS data (not started).
  2. The GNN crosstest (only BA data and checkpoints exist).
  3. An online selector or bandit.
  4. Evidence beyond MIS.
  5. Larger samples and error bars.

---

## 1. The research idea

**Base system.** This repo is a fork of LLM4Branch (`upstream: hzn18/LLM4Branch`),
which is itself built on openevolve. LLM4Branch uses an LLM to evolve a Python
`score_function` over 91 per-variable features. SCIP calls that function at every
B&B node to choose the branching variable. LLM4Branch ships one evolved policy per
problem family: setcover, cauctions, indset, facilities.

**Our question.** LLM4Branch evaluates each family on one generator setting. Real
instance streams vary *within* a family. Does one evolved policy per family hold up
under within-family structural variation? If not, does fixing it need new
*structure* (new features and logic), or just retuned *parameters*?

**Evidence: the "oracle gap."** For a combined test set spanning several regimes:
1. Compute the geomean solve time of each fixed policy used everywhere.
2. Compute the geomean of an oracle that picks the best fixed policy per instance.
3. The oracle gap is (best single policy's geomean) / (oracle's geomean).

A large gap shows no fixed policy is enough. It is also an upper bound on what a
selector could recover.

**Why LLM evolution, not just BO retuning.** BO only reweights an existing formula.
On RR, reweighting the released IndSet policy's 4-feature structure had *no*
exploitable signal: a near 50/50 coin flip. Only LLM-discovered *new features*
found real signal. Retuning those new structures on BA still left them about 22×
worse than Original. So specialization is structural, not parametric.

---

## 2. Environment and setup

**Machine.** Apple Silicon Mac. All numbers were produced locally, usually with
6–8 worker processes.

| Component | Version / location |
|---|---|
| Python | 3.10.18, venv at `~/Desktop/scip-build/venv` |
| SCIP | 8.0.0, **built from source**, installed to `~/Desktop/scip-build/install` (`libscip.8.0.dylib`) |
| PySCIPOpt | pip 6.2.1 (reports `__version__` 4.3.0 internally), linked against the SCIP above |
| ecole | **custom fork** `github.com/hzn18/LLM4Branch-ecole` @ `debd0e44dc5c5c200fcb3845d511cd0729baaa3b`, reports 0.8.1. Stock PyPI ecole will not work (custom observation functions). |
| numpy | 2.2.6 |
| scikit-optimize | 0.10.2 (BO tuning) |
| networkx | 3.4.2 (RR/WS generators) |
| lightgbm | 4.7.0 (GBDT baseline) |
| torch | 2.13.0 (MLP/GNN baselines; use CPU, see pitfalls) |
| boto3 | needed for the Bedrock LLM backend |

**Shell setup used for every run:**
```bash
source ~/Desktop/scip-build/venv/bin/activate
export DYLD_LIBRARY_PATH=$HOME/Desktop/scip-build/install/lib:$DYLD_LIBRARY_PATH
cd <repo root>
export PYTHONPATH=$(pwd)
```

**Install this repo:** `pip install -e .` from the repo root. That installs the
`openevolve` package, plus `openevolve-run.py`.

**LLM access.** We used DeepSeek-R1 via AWS Bedrock, to match the paper's stated
method ("powered by DeepSeek-R1").
- `openevolve/llm/bedrock.py` is new: a Bedrock Converse API client.
- `openevolve/llm/ensemble.py` is modified. It routes any model whose `api_base`
  starts with `bedrock://<region>` to `BedrockLLM`, and everything else to the
  OpenAI-compatible client.
- Auth uses the bearer token in env var `AWS_BEARER_TOKEN_BEDROCK`. We kept it in
  `.env`, which is gitignored and **not committed**.
- The model id is `us.deepseek.r1-v1:0` (see the `llm:` block in `config_rr.yaml`
  and similar files).
- `configs/my_key_set.yaml` (gitignored) holds API keys for non-Bedrock backends.
  `configs/key_set.yaml` is the template.

**Large artifacts NOT in the git repo.** They are about 9 GB and can be regenerated
with the scripts in §5.
- `examples/brancher/baselines/*/data*/`: imitation-learning datasets.
- `examples/brancher/baselines/MLP/_quarantine/`.

---

## 3. Code map — what each file is

Paths are relative to the repo root. Experiment scripts add their own directory and
`examples/brancher/` to `sys.path`. **Run everything from the repo root with
`PYTHONPATH=$(pwd)`.** `tester_parallel.py` writes CSVs to
`./examples/brancher/result/...`, relative to the cwd.

### 3.1 Upstream framework (mostly unchanged)
| Path | What it is |
|---|---|
| `openevolve/` | The openevolve evolutionary framework: islands, MAP-Elites database, prompt building, LLM ensemble, cascade evaluation. Local changes: `llm/ensemble.py` (backend routing) and `llm/bedrock.py` (new). |
| `openevolve-run.py`, `openevolve/cli.py` | Standard CLI launcher. **Has a hardcoded dataset whitelist** (`setcover/cauctions/indset/facilities/file/user-define`), so our custom domains use their own launchers (`run_*_evolution*.py`). |
| `configs/` | Default openevolve configs plus the key-set template. |
| `README.md` | Upstream LLM4Branch README. |

### 3.2 `examples/brancher/` — the LLM4Branch branching example
| Path | What it is |
|---|---|
| `utils/utils.py` | Core helpers. `DATASET_PARAMS` holds the easy/medium/hard tiers for the 4 built-in families. `create_instance(dataset, seed, level)` wraps the ecole generators. `get_scip_params()` sets SCIP params (see §4.3). `load_program()` imports a policy file. `normalize()` min-max normalizes features per node. `geometric_mean_stable()` is the shifted geomean `exp(mean(log(x+1)))-1`. |
| `tester_parallel.py` | **Main evaluation harness.** `solver()` runs one instance with either `relpscost` (RPB) or `evolve` (a policy file) through an ecole branching env. `evaluate()` fans seeds out over a process pool pinned to CPU cores via lock files. CLI: `--program_path --dataset --easy/--medium/--hard --methods relpscost,evolve --max_workers --cores_list --lock_dir --csv`. Default test seeds start at `999999999`; `instance_num=80`. Locally modified relative to upstream (see `git diff` against upstream). **Swallows exceptions; see pitfalls.** |
| `tester.py` | Older single-process tester. |
| `evaluator.py` | openevolve evaluator for the 4 built-in families (stage1/stage2 cascade, see §4.2). |
| `evaluator_config.yaml` | Evaluator settings. `target: nodes`, `train_seed: 666666`, `train_num: 4`, `valid_seed: 666666`, `valid_num: 8`, plus per-domain timeouts and BO budgets (`n_calls: 50, n_initial: 10` for indset/rr/ws). We added `rr` and `ws` blocks. |
| `config.yaml` | Upstream evolution config (SetCover). |
| `config_category_g.yaml` | Evolution config for SetCover Category G (Aug 30). |
| `config_rr.yaml`, `config_rr_smoke.yaml` | Evolution config for RR. DeepSeek-R1 via Bedrock, 50 iterations, cascade thresholds `[0.85, 0.75]`, population 40, 4 islands. The system prompt includes the full 91-feature table and RR-specific hints. |
| `config_ws.yaml`, `config_ws_smoke.yaml` | Same as above, for WS. |
| `config_specialist.yaml` | Same as above, for the BA worst-decile specialist (Sep 16, not yet run). |
| `initial_program.py` | Generic seed program: one feature, index 43 (pseudocost product). |
| `program/{setcover,cauctions,indset,facilities}/program.py` | **LLM4Branch's released policies.** "Original" in our tables = `program/indset/program.py`. It uses 4 features: `[0 obj_coeff, 7 reduced_cost, 9 fractional_part, 22 constraint_count]` and `PARAMS ≈ [1.079, 0.238, 1.222, 0.946]`, bounds `[0,2]`. |
| `generator.py` | Upstream imitation-data generator (explore-then-strong-branch). |
| `scripts/run.sh`, `eval-parallel.sh`, etc. | Upstream launch scripts, written for a 64-core server. Adjust `CORES`. |
| `result/` | CSVs written by `tester_parallel.py --csv`, mostly SetCover tiers and D_sub holdouts. |
| `log/eval/`, `log/experiments/` | Run logs for nearly every experiment. Filenames match the scripts. |
| `tmp/cpu_locks/` | Core-lock files (runtime only). |

### 3.3 `examples/brancher/experiments/` — our research
| Folder | Phase | Contents |
|---|---|---|
| `sensitivity/` | Aug 27–28, SetCover | Sensitivity of the released SetCover policy: `coordinate_sweep.py` (one param at a time), `random_perturbation.py` (all 7 params at radius r), `worst_case_large_radius.py`. Result CSVs are alongside. |
| `zo_seed_variance/` | Aug 28, SetCover | Re-ran LLM4Branch's BO param tuning with 10 random seeds (`program_zoseed{1..10}.py`), then evaluated on 80 held-out medium instances (`holdout_results/`). |
| `dsub_spread/` | Aug 27–29, SetCover | `tune_dsub.py` re-tunes with tuning-set size D_sub ∈ {4, 40, 100} (nested seeds from 666666). `n_calls_convergence.py` extends one BO run to 50/100/150/200 calls. Programs and holdout CSVs are alongside. |
| `structural_generalization/` | Aug 29 → Sep 16 | **Everything else.** ~200 files, indexed in §3.4. |

### 3.4 `structural_generalization/` index (grouped by purpose)

**SetCover structural categories (Aug 29–31).** Pattern: `category_X_generator.py`
builds instances, `eval_category_X.py` runs RPB vs Original, and the results go to
`category_X_results.csv`.
- A / A-intense: heavy-tailed column degree plus bottleneck rows.
- B: clustered/modular.
- B2 (`b_decoy`), B3 (`b3_decoy`): B plus decoy bridge columns.
- C: cost–coverage correlation.
- E: exact column symmetry.
- F: hybrid of A and B.
- G: B plus multi-cover rows (RHS ≥ 2).
- H: geometric disk covering.
- I: odd cycles (failed design).
- J / J2 / J3: inverted constraint-importance trap.

Other files in this group:
- `tune_category_{g,j,j2,j3}.py` re-tune the released 7-param SetCover formula on a
  category (skopt `gp_hedge`, `random_state=123`, target mean nodes, D_sub = seeds
  0–3). Output: `program_category_*_tuned*.py`.
- `eval_tuned_only.py`, `tuned_results_*.csv` and `standard_comparison/*.csv` compare
  the tuned arms against Original on standard SetCover medium.
- `eval_arm_standard_fresh.py`, `eval_arm_categoryg_fresh.py` and
  `scip_build_scripts/run_fresh_validation.sh` plus `compute_full_multiarm.py` form
  the 150+150 fresh-seed, 6-arm validation driver. It crashed; see the pitfalls.
- `curated_hard_seeds_{easy,medium}.txt`.

**IndSet Category K (Aug 31) — BA affinity and Erdős–Rényi:**
- `category_k_indset_affinity.py` builds BA at a chosen affinity. **This is the BA
  generator used for all BA training tiers later.**
- `category_k_er_indset{,_large}.py` builds ER graphs.
- `calibrate_k*.py` are calibration probes.
- `tune_category_k{,_ba}.py` produce `program_category_k_*tuned*.py`.
- `validate_k_tuning_harness.py`, `run_rpb_k.py`.

**RR phases (Aug 31).** These refer to a `regular.md` that is *not in the repo*.
- `category_rr_indset.py` is the **RR generator**.
- `calibrate_rr_phase1*.py`, `calibrate_rr_20s.py` are calibration probes.
- `tune_category_rr.py` produces `program_category_rr_tuned.py` (BO on Original's
  structure: "Tuned-via-RR").
- `crosstest_rr_phase{2,4}.py`.
- `phase5_sobol_landscape.py`: 128 Sobol param vectors evaluated on both BA and RR.
- `phase6_feature_contributions.py`: per-term score spread at real B&B nodes.
- `verify_dead_features.py`: obj_coeff and reduced_cost are constant across
  candidates for unweighted MIS.
- `phase8_grid_scan.py`, `phase8_lambda_scan.py`, `phase8b_paired_confirm.py`:
  fractionality-vs-degree tradeoff scans.

**RR and WS evolution (Sep 1–2), the core result:**
- `evaluator_rr.py` / `evaluator_ws.py`: openevolve evaluators wired to the custom
  generators.
- `run_{rr,ws}_evolution{,_smoke}.py`: programmatic launchers that bypass the CLI
  whitelist.
- Output folders `{rr,ws}_evolution_output/`: `checkpoints/checkpoint_N/`, `best/`
  and `top/`.
- `crosstest_evolved_rr.py`, `crosstest_rr_n250_rerun.py`,
  `crosstest_evolved_rr_ba_tuned.py`.
- `tune_evolved_{rr,ws}_on_ba.py` produce `program_evolved_{rr,ws}_ba_tuned.py`
  (BA-specific BO retune of each evolved structure).
- `category_ws_indset.py` is the **WS generator**. `calibrate_ws{,2,3,4}.py`
  calibrate it.
- `screen_ws.py`, `screen_evolved_ws.py` are cheap 4-seed screens vs RPB.
- `tune_original_on_ws.py`, `eval_ws_testset.py`, `eval_evolved_ws_on_rr.py`.
- `compute_oracle_table.py` builds the final oracle table by parsing per-instance
  times from logs.

**RPB tuning (Sep 5–6):**
- `tune_rpb.py` BO-tunes SCIP relpscost knobs (`pscostweight`, `conflictweight`,
  `inferenceweight`, `cutoffweight`, `minreliable`) per domain. Outputs
  `rpb_tuned_{ba,rr,ws,mixed,default}{,_run2,_run3,_bigscale}.json`.
- `crosstest_rpb.py` runs a 3×3 (×5 tunings) crosstest →
  `crosstest_rpb_results{,_run2,_run3,_bigscale}.json`.
- `crosstest_rpb_recalibrated.py` reruns on the harder RR/WS tiers →
  `crosstest_rpb_results_recalibrated.json`.

**Difficulty sweep and BA-matched re-evolution (Sep 9–11):**
- `sweep_ba_vs_rr_ws_difficulty.py` → `sweep_ba_vs_rr_ws_results.json`. The
  write-up is `ba_gap_report.html`.
- `calibrate_sweep_checkpoints.py`, `calibrate_ws_knob_sweep.py`.
- `run_{rr,ws}_evolution_extend.py` extend the original runs from 50 to 200
  iterations. **This overwrote `best/`; see the pitfalls.**
- `ws_on_ws_rerun_results.json`.
- `rpb_train_difficulty_check.py`.
- `evaluator_{rr,ws}_batrain.py`: BA-difficulty-matched evaluators (RR n=316, WS
  n=300) using the **forkserver** nested pool.
- `run_{rr,ws}_evolution_batrain.py` → `{rr,ws}_evolution_batrain_output/`.
- `cheap_tune_test.py` → `cheap_tune_test_result.json`.
- `test_batrain_policies_hard_tier.py`.
- `run_ws_evolution_batrain_warmstart.py` → `ws_evolution_batrain_warmstart_output/`.
- `test_ws_warmstart_hard_tier.py`, `test_ws_on_ba.py`, `test_ba_on_ba.py`.

**BA worst-decile analysis (Sep 12):**
- `worst_decile_analysis{,_extend,_extend2}.py` →
  `worst_decile_analysis_results.json` (1,300 BA n=1000 instances, policy vs RPB,
  graph statistics).
- `worst_decile_compare.py`.
- `worst_decile_fractionality{,_compare}.py`.
- `worst_decile_clique_{stats,compare}.py`.
- `worst_decile_score_{discrimination,compare}.py`.
- `worst_decile_hub_redundancy.py`.
- `worst_decile_early_signal{,_v2}.py`, `extend_early_signal.py`.
- Each has a matching `*_results.json`.
- Histograms: `ratio_histogram.png` and `ratio_histogram_1300.png` at the repo root.

**BA worst-decile specialist (Sep 15–16, not run):**
- `calibrate_specialist_training_size.py` → `calibrate_n300_results.json`,
  `calibrate_n750{,_pool2,_combined}_results.json`.
- `specialist_training_set_n750.json`: 4 losers and 4 worst-decile non-losers.
- `evaluator_specialist.py`, `run_specialist_evolution.py`, `config_specialist.yaml`.
  The output dir `specialist_evolution_output/` does not exist yet.

### 3.5 `examples/brancher/baselines/` — learned-policy baselines (Sep 2–8)
| Folder | Contents |
|---|---|
| `MLP/` | `model.py` (MLP over 91 features), `generate_data.py`, `train.py`, `select_checkpoint.py`, `crosstest.py` → `crosstest_mlp_results.json`. Final per-domain models: `mlp_{ba,rr,ws}.pt`. Epoch checkpoints: `checkpoints_{ba,rr,ws}/`; `checkpoints/` is the older SetCover MLP. |
| `GBDT/` | LightGBM lambdarank. `generate_data.py`, `train.py`, `crosstest.py` → `crosstest_gbdt_results.json`. Models: `gbdt_{ba,rr,ws}.txt`. |
| `GNN/` | Gasse-style bipartite GCNN. `generate_data.py` (SetCover), `generate_data_ba.py`, `model.py`, `train.py`, `select_checkpoint.py`, `checkpoints_ba/`. **Only BA was trained; there is no RR/WS model and no crosstest.** |
| `/GNN_Code/` (repo root) | Standalone, cleaned-up GNN reproduction package for co-authors. It has its own README, `requirements.txt`, exact seeds and known issues. **Start there for the GNN baseline.** |

---

## 4. How the pieces work

### 4.1 A branching policy file
Every policy (released, evolved or tuned) is a Python file with this shape:
```python
USED_FEATURES = [...]        # indices into the 91-dim feature vector
PARAMS = [...]               # tunable weights (BO tunes these)
BOUNDS = [[lo, hi], ...]     # BO search box per param
def score_function(variable_features, params):  # (n_candidates, 91) -> (n_candidates,)
    ...
```

At each B&B node, ecole's `NodeBipartite` + `Khalil2016` observation gives 19 + 72 =
91 features per variable. The `USED_FEATURES` columns are min-max normalized across
candidates (`utils.normalize`), and SCIP branches on the argmax score.

The feature index table is in the `prompt.system_message` of `config_rr.yaml`.
- 0–18: NodeBipartite. 0 = obj coefficient, 7 = reduced cost, 9 = fractional part, …
- 19–90: Khalil2016:
  - 22: number of participating constraints
  - 37–38: fractional and ceiling distance
  - 39–43: pseudocosts
  - 48–54: dynamic constraint degrees and their ratios
  - 55–66: coefficient/RHS ratios
  - 67–90: active-constraint statistics under 4 weightings

### 4.2 Evolution loop (openevolve plus a two-stage cascade evaluator)
1. The LLM (DeepSeek-R1) proposes a new `score_function` program, using the
   population and the system prompt.
2. **Stage 1** (`evaluate_stage1`) solves `train_num = 4` training instances (seeds
   from `train_seed = 666666`) with the candidate and with RPB.
   `stage1_score = min over seeds of (rpb_time / policy_time)`, a worst-case
   speedup. It is **time-based**, so run it on a quiet machine.
3. If stage 1 clears the cascade threshold (0.85 in our configs), **stage 2**
   (`evaluate`) runs:
   - **BO-tunes PARAMS** within BOUNDS using skopt `gp_hedge`, `n_calls = 50`,
     `n_initial = 10`, objective = mean node count on the training seeds.
   - Validates on `valid_num = 8` seeds.
   - Returns `combined_score` and time/node metrics.
4. The best program goes to `<output>/best/best_program.py`, and a snapshot to
   every `checkpoints/checkpoint_N/` (every 10 iterations).

Our custom domains use `run_*_evolution*.py`. These build `Config` and `EvaConfig`
in Python and call `OpenEvolve` directly, because of the CLI whitelist.

### 4.3 SCIP settings (`get_scip_params`)
- `separating/maxrounds = 0` (no cuts after the root).
- `presolving/maxrestarts = 0`.
- `timing/clocktype = 1` (CPU time).
- `limits/time` = 3600s by default. Many experiments pass smaller caps (120s,
  180s, 900s); see §5.
- RPB = SCIP's `relpscost` with `branching/relpscost/priority = 9999999`.
- Reported "time" is SCIP process/solve time. Geomeans always use
  `geometric_mean_stable` (shift 1).

### 4.4 Methodology rules (fixed conventions; also in memory/`portfolio.md`)
1. **Train evolution on an easier tier than the test tier.** Never train on the
   test tier itself, but don't make training so easy that there is no gradient.
   A WS attempt at p=0.15 gave about 600 mean nodes and was rejected.
2. **Oracle-table rows are structures, not parameter vectors.** A row may use
   native params on one domain and a BA-specific BO retune on another.
3. **Native params first.** Retune a (row, domain) cell only if a 4-seed screen
   against RPB fails badly. Example: 0/4 optimal, or a projected ~10 h for the
   full test set. `speedup < 1` alone does not trigger a retune.
4. **Same seeds for every row within a domain.** This keeps the oracle's
   per-instance min valid and the comparisons paired.
5. **Pool raw per-instance times across domains, then call
   `geometric_mean_stable` once.** Never take a geomean of geomeans: the shifted
   formula does not decompose.
6. **Oracle** = per-instance min over the arms, pooled the same way.
7. Validation runs are **fully sequential across arms** for fair, contention-free
   timing.

---

## 5. Data and instance generation

### 5.1 Generators
| Domain | Generator | Formulation |
|---|---|---|
| SetCover (standard) | `ecole.instance.SetCoverGenerator(n_rows, n_cols, density=0.05)` via `utils.create_instance` | ecole |
| SetCover categories A–J | `category_*_generator.py` (custom numpy, then pyscipopt) | covering rows ≥ 1 (G: some ≥ 2) |
| **BA** MIS | `ecole.instance.IndependentSetGenerator(graph_type=barabasi_albert, n_nodes, affinity)`. Standard tiers via `utils.create_instance('indset', seed, level)`; other sizes/affinities via `category_k_indset_affinity.generate_instance` | **clique-cover formulation** (ecole emits clique constraints, not one per edge) |
| **RR** MIS | `category_rr_indset.generate_instance(seed, n_nodes, degree)` → `networkx.random_regular_graph(d, n, seed)`. If `n*d` is odd, n is bumped by 1. | **edge formulation** (`x_u + x_v ≤ 1` per edge) |
| **WS** MIS | `category_ws_indset.generate_instance(seed, n_nodes, k, p)` → `networkx.watts_strogatz_graph` | edge formulation |
| ER MIS (Cat. K) | `category_k_er_indset*.py` | edge formulation |

> **Caveat for interpreting features.** BA (ecole) uses a clique cover while RR/WS
> use per-edge constraints. So feature 22 ("number of participating constraints")
> is *not* literally vertex degree on BA, even though `portfolio.md` describes it
> that way. It is roughly how many cover cliques the vertex sits in. It still
> correlates with degree. Keep this in mind when writing up the "degree signal"
> story.

### 5.2 Tiers, seeds and time caps

**Main oracle-table test sets (30 instances per domain, same seeds for every row):**
| Domain | Test tier | Seeds | Cap |
|---|---|---|---|
| BA | n=1000, affinity=4 (`indset` medium) | 999999999 … 1000000028 | 3600s |
| RR | n=250, degree=3 | 1000 … 1029 | 120s |
| WS | n=200, k=20, p=0.25 | 200 … 229 | 120s |

These same tiers and seeds are used by the MLP, GBDT and original RPB crosstests.

**Harder or alternative test tiers:**
| Use | Tier | Seeds | Cap |
|---|---|---|---|
| RPB recalibrated (Sep 6) | RR n=310 deg 3; WS n=200 k=20 p=0.3 | same as above | — |
| Difficulty sweep, "hard tier" (Sep 9–11) | RR n=330 deg 3; WS n=250 k=20 p=0.25; BA n=1000 aff 4 | 800000000 … 800000029 | 900s |
| BA worst-decile population (Sep 12) | BA n=1000 aff 4 | 800000000 … 800001299 (1,300) | 900s |
| Specialist calibration (Sep 16) | BA n=750 aff 4 (also n=300 tried) | 900000000+ block | — |

**Training tiers** (always easier than test):
| Use | BA | RR | WS |
|---|---|---|---|
| Evolution (original lineage) | (released policy: n=750 aff 4) | n=200 deg 3 (geomean ≈ 2.58s, ~4,041 nodes) | n=200 k=20 p=0.2 (geomean ≈ 5.09s, ~3,798 nodes) |
| BO retunes on BA, GNN data | n=400 aff 5 | — | — |
| MLP/GBDT data, checkpoint selection | n=300 aff 5 | n=200 deg 3 | n=200 k=20 p=0.2 |
| "batrain" (BA-difficulty-matched) | n=750 aff 4 | n=316 deg 3 | n=300 k=20 p=0.2 |
| Specialist | n=750 aff 4, 8 curated seeds | — | — |
| RPB tuning | n=400 aff 5 | n=200 deg 3 | n=200 k=20 p=0.2 |

**Other seed conventions:**
- Evolution and BO training seeds start at 666666 (`train_num 4`, `valid_num 8`).
- Ad-hoc BO tunes use D_sub = seeds 0–3.
- WS calibration used seeds 100–103.
- Sobol and grid scans used 100–102 (BA) and 2000–2002 (RR).
- SetCover standard test: 80 instances from 999999999. The "fresh" validation
  block is 1000000079 … 1000000228.

### 5.3 Imitation-learning data (baselines)
- **Expert:** strong branching via ecole `StrongBranchingScores`.
- **Features:** the same 91 (NodeBipartite + Khalil2016) for MLP and GBDT. The
  GNN uses the bipartite graph (variable, constraint and edge features). The GNN
  stores graph structure once per instance (`graph_*.pkl`) and per-node dynamics
  per sample.
- **Episodes** are capped at 180s.

| Model | Dataset size | Seeds |
|---|---|---|
| GBDT | 15k train / 3k val per domain | train seeds 0+, val 100000000+ |
| MLP | target 100k / 20k per domain. Reuses the GBDT data plus new samples (`--num_samples 85000 --seed_start 50000000`; val `17000 @ 150000000`). | — |
| GNN (BA only) | 100k / 20k at BA n=400 aff 5 | — |

GNN data generation for BA took about 10 h with 6 workers.

**Commands (per domain d ∈ {ba, rr, ws}), run from `examples/brancher/baselines/<MODEL>/`:**
```bash
python generate_data.py --domain d --split train --num_samples N --seed_start S --num_workers 6
python generate_data.py --domain d --split val   --num_samples M --seed_start S2 --num_workers 6
python train.py --domain d [--device cpu]          # GBDT: single fit, no epochs
python select_checkpoint.py --domain d             # MLP/GNN: choose by real solve time, not loss
python crosstest.py ...                            # 3x3 matrix, resumes per cell
```
Check each script's `--help` for exact flags.

---

## 6. Full timeline and results

### Phase 0 — Aug 26: setup
- Forked LLM4Branch and got the SCIP 8 + custom ecole build working.
- Added the Bedrock/DeepSeek-R1 backend.
- Added CPU-lock-file core pinning.

### Phase 1 — Aug 27–29: how fragile is the released SetCover policy? (`sensitivity/`, `zo_seed_variance/`, `dsub_spread/`)

**Coordinate sweep** (medium tier, mean time):
- Only two of the 7 params matter much: params[6] (constraint_importance, range
  20.7–26.3s) and params[2] (pseudocost_strength, 21.1–23.7s).
- The other params move results by under 1s.
- Script notes say params[6] alone accounts for about 92.5% of the policy's margin.

**Random all-params perturbation:**
- Radius ≤ 0.2 gives small degradation. The directed worst case reached only about
  19.5% of the margin.
- `worst_case_large_radius.py` pushes both dominant params to 0.

**BO-seed variance:**
- We re-ran LLM4Branch's own BO tuning with 10 different random seeds.
- Held-out medium geomeans (wall time) ranged from **20.1s to 41.3s**, roughly a
  2× spread. The published tuning procedure is unstable.

**n_calls convergence:**
- 50, 100, 150 and 200 calls all give about 21–22s.
- So the instability is not under-exploration. It points to a multi-modal
  landscape.

**D_sub ∈ {4, 40, 100}:**
- Tunes are in `dsub_spread/`. Holdouts are in `result/setcover/*dsub*` and
  `log/eval/setcover_dsub_medium_*`. No write-up was found.

### Phase 2 — Aug 29–31: within-family SetCover structural categories
Goal: find SetCover sub-structures where the released policy loses to RPB.
Results are RPB vs Original geomean times; "RPB wins" counts instances.

| Category | Mechanism | n | RPB gm | Original gm | RPB wins |
|---|---|---|---|---|---|
| A | heavy-tailed degree + bottleneck rows | 30 | 6.15 | 3.22 | 0 |
| A-intense | more extreme A | 15 | 5.06 | 2.60 | 0 |
| B | clustered / modular | 25 | 12.78 | 10.83 | **7 (28%)** |
| B2 decoy | B + decoy bridges | 10 | 2.22 | 1.29 | 0 (made instances easier) |
| B3 decoy | calibrated B + surgical decoys | 10 | 6.35 | 3.84 | 0 |
| C | cost–coverage correlation | — | — | — | results CSV empty |
| E | exact column symmetry | 8 | 0.04 | 0.03 | 2 (trivial, ~1 node) |
| F | A + B hybrid | 8 | 0.17 | 0.16 | 3 (trivial) |
| G | B + multicover (RHS ≥ 2) | see below | | | |
| H | 2D disk covering | 10 | 2.24 | 1.17 | 0 |
| I | odd-cycle vertex cover | — | — | — | **design failed**: simplex returns a vertex with one fractional var per cycle, so it solves at the root |
| J / J2 / J3 | inverted constraint-importance trap | — | — | — | J: right direction but weak (−1.4% on J, +3.8% worse on standard). J2 overshot (mean nodes collapsed to ~2.5). J3 recalibrated and tuned (`program_category_j3_tuned.py`); no final result found. |

**Category G** was the most promising.
- The released 7-param formula was re-tuned on G's easy tier with BO seeds
  123, 1, 7 and 42 (`program_category_g_tuned*.py`).
- Every tuned variant still beat Original on standard SetCover medium, by
  different margins.
- There was a genuine rank reversal: Tuned-7 was best on standard and worst on G;
  Tuned-1 was the reverse.
- Multi-arm oracle gaps on 80+80 instances: **3.2%** (standard only), **5.1%**
  (G only), **7.5%** (mixed stream). Real, but small. See `backlog.md` §2.
- The planned **150+150 fresh-seed, 6-arm validation**
  (`scip_build_scripts/run_fresh_validation.sh`) **crashed after its first arm**
  (RPB on standard). It's a path bug; see the pitfalls. It was never re-run.

**Decision:** SetCover gaps were too small. We pivoted to Independent Set.
`backlog.md` notes that IndSet's BA generator is natively heavy-tailed.

### Phase 3 — Aug 31: IndSet, Category K (BA affinity, Erdős–Rényi)
- At the same average degree (~8), BA n=400 solves in a 0.83s geomean, while ER
  times out (>120s).
  - This is a degree-distribution *shape* effect, not a density effect.
- ER large tier calibrated at n=1000, p=0.0036 (geomean ≈ 5.8s).
- The released IndSet program was re-tuned on ER and on high-affinity BA
  (`program_category_k_*`).

### Phase 4 — Aug 31: RR (random-regular) diagnosis, phases 1–8
- RR calibrated at n=250, degree=3 (geomean ≈ 29s, all optimal).
- **Dead features:** obj_coeff (f0) and reduced_cost (f7) are constant across
  candidates for unweighted MIS on BA, RR and ER (`verify_dead_features.py`,
  15 configs, 30+ nodes deep).
- Sobol landscape (phase 5) and grid/λ scans (phase 8):
  - BA prefers λ ≈ 0.4 on the fractionality-vs-count tradeoff.
  - RR prefers λ ≈ 0.95–1.0 (pure fractionality).
  - The 8b paired confirmation used 50 instances per domain.
- **Original's structure has no exploitable preference on RR**: about 50/50 win
  rate across parameter extremes in a 50-seed paired test.
- BO retune on RR → "Tuned-via-RR" is degenerate.

### Phase 5 — Sep 1–2: LLM evolution finds structural specialists (core result; `portfolio.md` §4)

**Evolved-RR** (50 iterations, training tier n=200 deg 3; best found at iteration 6):
```
USED_FEATURES = [37, 42, 43, 48, 51]
score = p0*pseudocost_product*dynamic_ratio + p1*fractional_dist
      + p2*pseudocost_sum + p3*dynamic_mean
```
- On RR: 4.56–5.19s geomean.
- On BA: about **22× worse** than Original (569.26s, even after a BA-specific BO
  retune). The specialization is structural.
- BA+RR 60-instance oracle gap: **1.91×** (31/29 split).

**WS calibration:**
- Trivially easy at p < 0.1 (~1 node).
- p = 0.3 shows timeouts.
- Test tier fixed at **n=200, k=20, p=0.25**.

**WS cheap screen** (native params, vs RPB 16.00s):
- Original: 29.99s (0.53×).
- Evolved-RR: 51.30s (0.31×).
- Both kept native params. WS is a fresh challenge.

**Evolved-WS** (50 iterations, training tier p=0.2; best at iteration 39):
```
USED_FEATURES = [22, 37, 38, 43, 52, 54, 55, 85]   (6 terms; BO zeroed 3)
```
- On RR: 3.96s native vs RPB 3.76s, kept native.
- On BA: 0/4 optimal, so it was retuned (`program_evolved_ws_ba_tuned.py`,
  θ = [2.0, 0.0, 3.0, 0.6526, 1.7536, 2.0]).
- **WS test set:**
  - Evolved-WS: **11.73s**, 30/30 optimal.
  - Original: 25.38s.
  - Evolved-RR: 40.29s, 29/30 optimal.
  - Evolved-WS wins 29/30.

**Final oracle table (90 instances: 30 BA + 30 RR + 30 WS):**

Per-domain geomeans (seconds):

| Row | BA | RR | WS |
|---|---|---|---|
| Original | 25.49 | 16.37 | 25.38 |
| Evolved-RR | 569.26 (tuned) | 5.19 | 40.29 |
| Evolved-WS | 1022.22 (tuned) | 5.08 | 11.73 |

Pooled over all 90 instances:

| Row | Pooled geomean | Wins /90 |
|---|---|---|
| Original | **21.98** (best fixed) | 32 |
| Evolved-RR | 51.63 | 15 |
| Evolved-WS | 41.95 | 43 |
| **Oracle** | **11.20** → **gap 1.96×** | — |

### Phase 6 — Sep 2–8: baselines (the "mixed-model" ablation, partially done)

**Tuned RPB** (relpscost knobs, BO per domain; Sep 5–6). Original tiers, geomean
seconds. Rows are the domain the knobs were tuned on; columns are the test domain.

| Tuned on ↓ / Test → | BA | RR | WS |
|---|---|---|---|
| ba | 70.38 | 3.83 | 9.63 |
| rr | 101.66 | 3.57 | 9.22 |
| ws | 71.65 | 3.90 | 9.66 |
| mixed | 65.71 | 3.54 | 8.76 |
| default | 69.32 | 3.40 | 9.27 |

- Per-domain tuning of RPB gives essentially **no specialization**.
- Replicates (run2, run3) and a bigger budget (`_bigscale`) confirmed this.
- Because RPB was so fast on the original RR/WS tiers, those tiers were
  recalibrated to RR n=310 and WS p=0.3. There, RPB is about 20–25s on RR and
  24–33s on WS for every tuning.

**MLP**, per-domain imitation (rows = trained on, columns = tested on). Geomean
seconds, optimal count in brackets.

| Trained ↓ / Test → | BA | RR | WS |
|---|---|---|---|
| ba | 50.52 (30/30) | 5.38 | 9.15 |
| rr | **693.67 (17/30)** | 3.85 | 14.03 |
| ws | **330.44 (24/30)** | 4.05 | 7.04 |

**GBDT** (LightGBM lambdarank):

| Trained ↓ / Test → | BA | RR | WS |
|---|---|---|---|
| ba | 63.47 (29/30) | 9.24 | 16.82 |
| rr | 263.14 (28/30) | 4.26 | 17.42 |
| ws | 508.13 (22/30) | 7.86 | 12.57 |

**Readout:**
- Learned per-domain models show the **same off-domain collapse onto BA** as the
  evolved specialists.
- On their home domains, MLP beats the evolved programs on RR (3.85s vs 5.19s)
  and WS (7.04s vs 11.73s). It loses on BA (50.5s vs Original's 25.5s).
- **Not done:** a single model trained on *pooled* BA+RR+WS data. This is the
  ablation that tests "one more expressive model closes the gap." The GNN was
  only trained on BA, and there is no GNN crosstest.

### Phase 7 — Sep 9–11: is BA's policy actually robust? (`ba_gap_report.html`)

**Difficulty sweep** (hard tier, 900s cap, seeds 800000000+). These cells use the
**200-iteration** evolved programs.

| Cell | Easy tier gap | Hard tier |
|---|---|---|
| BA-policy on RR vs RR-policy | 3.15× | **9.19×** (382.76s vs 41.63s; BA policy 25/30 optimal) |
| BA-policy on WS vs WS-policy | 2.16× | **2.13×** (148.90s vs 69.86s), flat |
| BA-policy on BA (hard-tier seeds) | — | 19.18s |

- Raising n further (RR n=500) made everything time out.
- Raising WS's k at fixed n (`calibrate_ws_knob_sweep.py`) was hypothesis 1.
- Extending WS evolution to 200 iterations was hypothesis 2.
- **Conclusion:** BA's apparent robustness in the main tables is partly an
  artifact of easy off-domain tiers. Against RR it falls apart at hard tiers;
  against WS the gap stays flat.

**BA-matched re-evolution ("batrain").**
- Training tiers were calibrated so that *plain RPB* sees similar difficulty to
  BA's n=750 training tier (`rpb_train_difficulty_check.py`):
  - RR n=316: RPB 18.17s vs BA 21.48s.
  - WS n=300.
- New 200-iteration evolution runs gave best stage1 scores below 1:
  - RR: 0.597.
  - WS: 0.733.
  - WS warm-started from WS-batrain: 0.799.
- None ever cleared the 0.85 gate, so none were stage-2 tuned during evolution.
- `cheap_tune_test.py` BO-tuned them anyway. Stage-1 scores *fell* (RR 0.56 → 0.53,
  WS 0.72 → 0.61). Tuning 4–9 linear coefficients can't rescue them.
- **Evolution did not beat RPB at BA-matched difficulty.**

**Hard-tier tests** (30 seeds, 900s):

| Policy | Tested on | Geomean | Optimal |
|---|---|---|---|
| rr_batrain_tuned | RR | 52.20s | 30/30 |
| ws_batrain_tuned | WS | 72.08s | 30/30 |
| ws_batrain_warmstart | WS | **54.31s** | 30/30 (best WS) |
| ws_old (easy-trained) | BA | 313.80s | 19/30 |
| ws_new_warmstart | BA | 574.84s | **8/30** |

A better WS specialist transfers *worse* to BA.

### Phase 8 — Sep 12: BA worst-decile analysis (1,300 BA n=1000 instances)
**Ratio** = Original policy time / RPB time, per instance.

| Quantity | Value |
|---|---|
| Original geomean | 25.47s |
| RPB geomean | 74.41s |
| Median ratio | 0.33 |
| p90 ratio | 0.58 |
| p99 ratio | 1.00 |
| Max ratio | 1.82 |
| Losers (ratio > 1) | 14 / 1300 (~1.1%) |

Does anything predict the ratio? Spearman correlation with the ratio, recomputed
for this document:

| Predictor | Spearman ρ |
|---|---|
| Degree CV | 0.05 |
| Clustering coefficient | 0.04 |
| n_edges | 0.08 |
| Root LP gap | **0.31** |
| Live gap after 10 nodes | 0.30 |
| Live gap after 25–200 nodes | 0.32–0.33 |

- Hub redundancy (top-1 vs top-2 hub degree), root fractionality,
  score-discrimination statistics and clique statistics were also checked. See the
  respective `*_results.json`.
- **Graph structure does not explain the hard cases. The root and early gap
  weakly do.** This suggests a cheap early-search signal for routing between
  policies.

### Phase 9 — Sep 15–16: worst-decile specialist (set up, NOT run)
- **Calibration:** BA n=750 reproduces the n=1000 phenomenon.
  - Median ratio 0.356 vs 0.33.
  - Worst-decile cutoff 0.580 vs 0.576.
  - Loser rate 0.7–1.0% vs 1.08%.
  - Solves take seconds.
- **Training set** (`specialist_training_set_n750.json`): 4 losers (ratio > 1) and
  4 worst-decile non-losers, split 4 train / 4 valid.
- **Run:** warm-start from `program/indset/program.py`, 200 iterations, forkserver
  pool. Output goes to `specialist_evolution_output/`, which does not exist yet.
- To launch:
  ```bash
  python examples/brancher/experiments/structural_generalization/run_specialist_evolution.py
  ```
- **Then:** test the specialist on the n=1000 worst-decile seeds against
  Original, and form a BA-internal 2-arm oracle.

---

## 7. Pitfalls and bugs found (read before reproducing)

1. **`tester_parallel.py` swallows exceptions.** `solver()` catches everything
   but never sets `validity=False`. A broken policy logs a fake success at
   `Proc Time: [0.0]`. Always sanity-check node counts. This bit us once:
   `write_program()` stripped `import numpy as np` from
   `program_evolved_ws_ba_tuned.py`. Fixed in that file, not in the tester.
2. **`best/best_program.py` was overwritten.** The 50→200-iteration extensions
   (Sep 9) overwrote it.
   - The **oracle-table** programs (Sep 2) are in
     `{rr,ws}_evolution_output/checkpoints/checkpoint_50/best_program.py`:
     - RR: features [37, 42, 43, 48, 51].
     - WS: features [22, 37, 38, 43, 52, 54, 55, 85].
   - The current `best/` holds the 200-iteration programs used by the Sep 9+ sweep:
     - RR: iteration 197, features [37, 38, 43, 48, 54, 56, 86, 89].
     - WS: iteration 95, features [9, 22, 37, 38, 43, 52, 54, 85].
   - Scripts like `eval_ws_testset.py` point at `best/`. Re-running them now uses
     the *newer* programs.
3. **Nested `fork` pool hangs** at BA-matched difficulty. The cause is TBB inside
   libscip plus Accelerate/GCD through numpy. The batrain and specialist
   evaluators use `multiprocessing.get_context("forkserver")` and need
   `structural_generalization/` on `PYTHONPATH`.
4. **Fresh-validation crash.** `eval_arm_standard_fresh.py` reads
   `./result/setcover/medium/{method}.csv`, but `tester_parallel.py` writes to
   `./examples/brancher/result/...`. The driver `cd`s to the repo root, so the
   move fails after the RPB arm. Fix the `src` path before re-running.
5. **Stage-1 scoring is wall/CPU-time based.** Background load changes scores.
   Use the CPU lock dir (`--cores_list`, `--lock_dir`) and a quiet machine.
6. **openevolve CLI dataset whitelist.** Use the `run_*_evolution*.py` launchers
   for rr, ws and the specialist.
7. **`evaluator_config.yaml` has `valid_seed == train_seed == 666666`.** The 8
   validation seeds therefore include the 4 training seeds. This is inherited
   from upstream; keep it in mind when interpreting stage-2 metrics.
8. **MPS (Apple GPU) training stalls for hours.** Train MLP/GNN with
   `--device cpu`. Training without `--seed` is nondeterministic.
9. **Strong-branching data generation at full test size is intractable.** That's
   why training tiers are small and episodes are capped at 180s.
10. **BA feature 22 is clique-membership count, not degree** (see §5.1).
11. Docstrings reference a `regular.md` design note (the RR investigation plan)
    that is **not in the repo**.
12. The Category-C results CSV is empty. Categories E and F are trivially easy, so
    they are not evidence either way.

---

## 8. Reproducing the key results (commands)

All commands run from the repo root after the §2 setup.

```bash
SG=examples/brancher/experiments/structural_generalization

# Released policy vs RPB on a built-in family/tier (writes examples/brancher/result/...)
python examples/brancher/tester_parallel.py --program_path examples/brancher/program/indset/program.py \
  --dataset indset --medium --methods relpscost,evolve --max_workers 6 --csv

# Evolve a specialist (RR shown; ws / *_batrain / specialist are analogous)
python $SG/run_rr_evolution_smoke.py      # 2-iteration smoke test first
python $SG/run_rr_evolution.py            # 50 iterations -> $SG/rr_evolution_output/

# Cheap screen and BA retune of an evolved structure
python $SG/screen_evolved_ws.py
python $SG/tune_evolved_ws_on_ba.py       # -> program_evolved_ws_ba_tuned.py

# WS held-out test set and the final oracle table
python $SG/eval_ws_testset.py             # NOTE: points at best/ (see pitfall 2)
python $SG/compute_oracle_table.py        # parses per-instance times from logs

# RPB tuning and crosstests
python $SG/tune_rpb.py --domain ba --n_calls 50 --n_initial 10   # rr, ws likewise
python $SG/crosstest_rpb.py
python $SG/crosstest_rpb_recalibrated.py

# Difficulty sweep / BA gap
python $SG/sweep_ba_vs_rr_ws_difficulty.py

# BA worst-decile population + early signal
python $SG/worst_decile_analysis.py
python $SG/worst_decile_early_signal_v2.py

# Next step
python $SG/run_specialist_evolution.py
```

- Most crosstest and sweep scripts save incrementally to JSON and skip finished
  cells on rerun.
- Expected runtimes: a 30-instance BA cell at a 3600s cap can take hours for a bad
  policy. A Evolved-WS-on-BA run took about 5h47m.
- A 50-iteration evolution run takes many hours. A 200-iteration run takes about
  a day.

---

## 9. Open items and next steps (priority order)

1. **Run the BA worst-decile specialist** (Phase 9). Then evaluate a BA-internal
   portfolio (Original + Specialist) on the 1,300-seed population. Routing could
   use the root/early-gap signal from Phase 8.
2. **Pooled single-model baseline.** Train one MLP, GBDT or GNN on BA+RR+WS
   combined. Does it close the 1.96× oracle gap? If yes, rethink the portfolio
   framing.
3. **Finish the GNN** (RR and WS data and training, then the 3×3 crosstest) using
   `GNN_Code/`.
4. **Fold the baselines into the oracle table.** Add MLP and GBDT rows and
   recompute the oracle over all arms on the same 90 seeds.
5. **Selector / online learning** (`portfolio.md` §5). The logged 90×k payoff
   matrix is a free full-information bandit environment. Simulate LinUCB or
   Thompson sampling with drifting mixtures.
6. **Generalize beyond MIS.** Options: the cross-family stream (`backlog.md` §0),
   or re-running the categorical search on another family.
7. **Statistics.** More instances per domain and bootstrap CIs on geomeans and
   oracle gaps.
8. Re-run the Category-G fresh validation after fixing pitfall 4, if the SetCover
   evidence is still wanted.
9. Update `portfolio.md` with Phases 6–9.
