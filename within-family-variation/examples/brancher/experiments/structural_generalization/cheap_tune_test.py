"""
Cheap test: does BO-tuning the (never-stage-2'd) batrain best policies lift them
past RPB, or is being slower-than-RPB structural at BA-matched difficulty?

For RR (n=316) and WS (n=300), on the TRAINING tier the policy was evolved at:
  1. stage1_score with ORIGINAL params        (baseline; should match best_program_info.json)
  2. evaluate() -> params_optimizer BO-tunes PARAMS (node objective, n_calls=50),
     returns tuned params + stage2 metrics (avg_time, avg_rpb_time, combined_score)
     on the 8 validation seeds
  3. stage1_score re-scored with the TUNED params
Report all three + per-seed. Interpretation:
  tuned stage1_score >= 1.0  -> missing stage-2/tuning was the problem; a lower-gate
                               re-run is justified
  tuned stage1_score  < 1.0  -> tuning 4 linear coeffs can't rescue it; "evolution
                               doesn't beat RPB at BA-matched difficulty" stands
Run only on a QUIET machine (stage1 re-score is time-based).
"""
import os
import re
import sys
import time
import json
import tempfile

SG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SG)
sys.path.insert(0, os.path.join(SG, "..", ".."))

import evaluator_rr_batrain as ev_rr
import evaluator_ws_batrain as ev_ws


class Cfg(dict):
    """dict with attribute access -- mirrors EvaConfig (needs both cfg.x and cfg.get(x,d))."""
    __getattr__ = dict.__getitem__


def make_cfg(lock_dir):
    os.makedirs(lock_dir, exist_ok=True)
    return Cfg(
        train_seed=666666, train_num=4,
        valid_seed=666666, valid_num=8,
        max_workers=8, cores_list=[0, 1, 2, 3], lock_dir=lock_dir,
        target="nodes",
        rpb=True,               # force stage2 to also solve RPB so we get avg_rpb_time
        parameter_opt=True,
        stage1_timeout=8000, stage2_timeout=10000,
        n_calls=50, n_initial=10,
    )


def swap_params(program_path, new_params):
    src = open(program_path).read()
    new_line = "PARAMS = [" + ", ".join(f"{v:.6g}" for v in new_params) + "]"
    src2 = re.sub(r"PARAMS\s*=\s*\[.*?\]", new_line, src, count=1)
    fd, tmp = tempfile.mkstemp(suffix="_tuned.py", dir=SG)
    os.write(fd, src2.encode())
    os.close(fd)
    return tmp


def run_one(name, ev, best_path, lock_dir):
    print(f"\n{'='*70}\n{name}: {best_path}\n{'='*70}", flush=True)
    orig_params = None
    m = re.search(r"PARAMS\s*=\s*\[(.*?)\]", open(best_path).read())
    if m:
        orig_params = [float(x) for x in m.group(1).split(",")]
    print(f"original PARAMS: {orig_params}", flush=True)

    cfg = make_cfg(lock_dir)

    t0 = time.time()
    s1_orig = ev.evaluate_stage1(best_path, cfg)
    print(f"[1] stage1 (ORIGINAL params): {s1_orig}   ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    full = ev.evaluate(best_path, cfg)      # runs params_optimizer + stage2
    tuned = full.get("params")
    print(f"[2] evaluate() done ({time.time()-t0:.0f}s)", flush=True)
    print(f"    tuned PARAMS: {tuned}", flush=True)
    for k in ("validity", "avg_time", "mean_time", "avg_rpb_time", "max_rpb_time",
              "avg_nodes", "avg_rpb_nodes", "avg_gap", "combined_score"):
        if k in full:
            print(f"    {k}: {full[k]}", flush=True)
    if full.get("avg_rpb_time") and full.get("avg_time"):
        print(f"    --> valid-seed avg speedup vs RPB (tuned): "
              f"{full['avg_rpb_time']/full['avg_time']:.3f}x", flush=True)

    s1_tuned = None
    if tuned:
        tmp = swap_params(best_path, tuned)
        try:
            t0 = time.time()
            s1_tuned = ev.evaluate_stage1(tmp, cfg)
            print(f"[3] stage1 (TUNED params): {s1_tuned}   ({time.time()-t0:.0f}s)", flush=True)
        finally:
            os.unlink(tmp)

    return {"name": name, "orig_params": orig_params, "tuned_params": tuned,
            "stage1_orig": s1_orig, "stage1_tuned": s1_tuned,
            "stage2_full": {k: full[k] for k in full if k != "params"}}


if __name__ == "__main__":
    results = []
    results.append(run_one(
        "RR n=316", ev_rr,
        os.path.join(SG, "rr_evolution_batrain_output", "best", "best_program.py"),
        os.path.join(SG, "..", "..", "tmp", "cpu_locks", "cheap_tune_rr")))
    results.append(run_one(
        "WS n=300", ev_ws,
        os.path.join(SG, "ws_evolution_batrain_output", "best", "best_program.py"),
        os.path.join(SG, "..", "..", "tmp", "cpu_locks", "cheap_tune_ws")))

    out = os.path.join(SG, "cheap_tune_test_result.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}", flush=True)
    for r in results:
        so = r["stage1_orig"].get("stage1_score") if isinstance(r["stage1_orig"], dict) else None
        st = r["stage1_tuned"].get("stage1_score") if isinstance(r["stage1_tuned"], dict) else None
        print(f"{r['name']}: stage1 orig={so}  ->  tuned={st}   "
              f"(>=1.0 means beats RPB worst-case)", flush=True)
    print(f"\nwrote {out}", flush=True)
