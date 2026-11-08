"""
Difficulty sweep: does BA-evolved's mild edge-deficit vs RR/WS's own
evolved policy (seen at the standard test tier in the main MLP crosstest --
ba-on-rr=5.38s vs rr-on-rr=3.85s, ba-on-ws=9.15s vs ws-on-ws=7.04s, both
still 30/30 optimal) grow into something like the catastrophic reverse
direction (rr-on-ba=693.67s/17-30 optimal, ws-on-ba=330.44s/24-30 optimal,
vs ba-on-ba's 50.52s/30-30) as RR/WS instances get harder -- or does it
stay proportionally mild? The comparison that matters is the RATIO/gap
between ba-on-X and X-on-X at each checkpoint, not absolute size-matching
to BA's own scale.

Uses the LLM-evolved programs (arbitrary Python scoring code over the
full feature vector, not a fixed-formula reweighting like RPB) since
those are structurally capable of specializing, unlike RPB.

Checkpoint 0 (standard test tier: RR n=250,degree=3 | WS n=200,k=20,p=0.25)
is NOT re-run here -- already measured in the main MLP crosstest, reused
for free as the sweep's baseline point.

2 new checkpoints, scaling n_nodes only (degree/k/p held at the
established convention):
  checkpoint 1: RR n=500,degree=3   | WS n=400,k=20,p=0.25
  checkpoint 2: RR n=1000,degree=3  | WS n=800,k=20,p=0.25  (RR reaches
                                       BA's own n_nodes here)

4 cells per checkpoint: ba-on-rr, rr-on-rr, ba-on-ws, ws-on-ws.
30 seeds per cell (same convention as the main 3x3 crosstest).

Same incremental-save/resume JSON pattern as crosstest.py.
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, get_scip_params, geometric_mean_stable, normalize
from category_rr_indset import generate_instance as generate_rr_instance
from category_ws_indset import generate_instance as generate_ws_instance

SG = os.path.dirname(__file__)
BA_PROGRAM = os.path.join(SG, "..", "..", "program", "indset", "program.py")
RR_PROGRAM = os.path.join(SG, "rr_evolution_output", "best", "best_program.py")
WS_PROGRAM = os.path.join(SG, "ws_evolution_output", "best", "best_program.py")
PROGRAMS = {"ba": BA_PROGRAM, "rr": RR_PROGRAM, "ws": WS_PROGRAM}

SEEDS = list(range(800000000, 800000030))  # fresh seed block, unused elsewhere in this project

# Calibrated via calibrate_sweep_checkpoints.py: each domain's OWN evolved
# policy solves comfortably here (RR n=330 -> 48.27s geomean all-optimal,
# degrades badly by n=360 i.e. 2/3 timeout; WS n=250 -> 50.46s geomean
# all-optimal, degrades badly by n=280). The originally-planned n=500/
# n=1000 checkpoints were a massive overshoot -- BOTH ba-trained AND each
# domain's own policy timed out completely there (0/2 optimal each in a
# smoke test), which isn't informative: no gap to measure if everything
# fails. This single checkpoint sits right at the edge of what each
# domain's own policy can still handle -- the most informative zone for
# the ba-vs-own comparison.
CHECKPOINTS = {
    1: {"rr": {"n_nodes": 330, "degree": 3}, "ws": {"n_nodes": 250, "k": 20, "p": 0.25}, "workers": 6},
}
TIME_LIMIT = 900  # 15 min cap per instance -- ~11-18x the calibrated working geomean (48-50s),
                  # ~4.5-5x the calibrated breaking point's timeout (180-200s)

RESULTS_PATH = os.path.join(SG, "sweep_ba_vs_rr_ws_results.json")


def make_instance(test_domain, checkpoint, seed):
    params = CHECKPOINTS[checkpoint][test_domain]
    if test_domain == "rr":
        return generate_rr_instance(seed=seed, n_nodes=params["n_nodes"], degree=params["degree"])
    else:
        return generate_ws_instance(seed=seed, n_nodes=params["n_nodes"], k=params["k"], p=params["p"])


def solve_one(args):
    import ecole
    import time as _time
    trained_on, test_domain, checkpoint, seed = args
    tag = f"[ckpt{checkpoint} {trained_on}-on-{test_domain} seed={seed}]"
    print(f"{tag} starting (cap={TIME_LIMIT}s)", flush=True)

    score_function, used_features, function_params, _ = load_program(PROGRAMS[trained_on])
    score_function = partial(score_function, params=function_params)

    flash_khalil_obs = ecole.observation.Khalil2016Flash()
    flash_node_obs = ecole.observation.NodeBipartiteFlash()
    flash_khalil_obs.set_active_features([f - 19 for f in used_features if f >= 19])
    flash_node_obs.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(
        observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
        scip_params=get_scip_params(time_limit=TIME_LIMIT),
    )
    env.seed(0)

    instance = make_instance(test_domain, checkpoint, seed)
    t0 = _time.process_time()
    observation, action_set, _, done, info = env.reset(instance)
    while not done:
        variable_features = __import__("numpy").concatenate(
            (observation["Node"].variable_features[action_set, :],
             observation["Khalil"].features[action_set, :]), axis=1)
        variable_features = __import__("numpy").nan_to_num(variable_features, nan=0.0, posinf=0.0, neginf=0.0)
        variable_features = normalize(variable_features, used_features)
        scores = score_function(variable_features)
        action = action_set[scores.argmax()]
        observation, action_set, _, done, info = env.step(action)
    proc_time = _time.process_time() - t0
    m = env.model.as_pyscipopt()
    status = m.getStatus()
    print(f"{tag} done: time={proc_time:.2f}s nodes={m.getNNodes()} status={status}", flush=True)
    return seed, proc_time, m.getNNodes(), status


def evaluate_cell(trained_on, test_domain, checkpoint):
    workers = CHECKPOINTS[checkpoint]["workers"]
    tasks = [(trained_on, test_domain, checkpoint, seed) for seed in SEEDS]
    with Pool(processes=workers) as pool:
        results = pool.map(solve_one, tasks)
    results.sort()
    times = [t for _, t, _, _ in results]
    n_optimal = sum(1 for *_, s in results if s == "optimal")
    return times, n_optimal


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_cell_result(checkpoint, trained_on, test_domain, times, n_optimal):
    results = load_results()
    key = str(checkpoint)
    results.setdefault(key, {}).setdefault(trained_on, {})[test_domain] = {
        "times": times, "n_optimal": n_optimal,
    }
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    existing = load_results()
    print("=== BA-vs-own-domain difficulty sweep (checkpoint 0 = existing crosstest data, not re-run) ===",
          flush=True)
    for checkpoint in args.checkpoints:
        params = CHECKPOINTS[checkpoint]
        print(f"\n--- checkpoint {checkpoint}: RR n={params['rr']['n_nodes']} | "
              f"WS n={params['ws']['n_nodes']} (workers={params['workers']}) ---", flush=True)
        for trained_on, test_domain in [("ba", "rr"), ("rr", "rr"), ("ba", "ws"), ("ws", "ws")]:
            key = str(checkpoint)
            if key in existing and trained_on in existing[key] and test_domain in existing[key][trained_on]:
                times = existing[key][trained_on][test_domain]["times"]
                n_optimal = existing[key][trained_on][test_domain]["n_optimal"]
                print(f"[ckpt {checkpoint}] {trained_on}-on-{test_domain}: [resumed] "
                      f"geomean={geometric_mean_stable(times):.2f}s optimal={n_optimal}/{len(times)}", flush=True)
                continue
            times, n_optimal = evaluate_cell(trained_on, test_domain, checkpoint)
            geo = geometric_mean_stable(times)
            save_cell_result(checkpoint, trained_on, test_domain, times, n_optimal)
            print(f"[ckpt {checkpoint}] {trained_on}-on-{test_domain}: geomean={geo:.2f}s "
                  f"optimal={n_optimal}/{len(times)}", flush=True)

    print("\n=== Gap summary (ba-on-X vs X-on-X) ===", flush=True)
    final = load_results()
    print("checkpoint 0 (existing, from main crosstest): "
          "ba-on-rr=5.38s(30/30) vs rr-on-rr=3.85s(30/30) -- ratio=1.40x | "
          "ba-on-ws=9.15s(30/30) vs ws-on-ws=7.04s(30/30) -- ratio=1.30x", flush=True)
    for checkpoint in sorted(int(k) for k in final.keys()):
        key = str(checkpoint)
        for test_domain in ["rr", "ws"]:
            if "ba" in final[key] and test_domain in final[key]["ba"] and \
               test_domain in final[key] and test_domain in final[key][test_domain]:
                ba_times = final[key]["ba"][test_domain]["times"]
                own_times = final[key][test_domain][test_domain]["times"]
                ba_geo = geometric_mean_stable(ba_times)
                own_geo = geometric_mean_stable(own_times)
                ba_opt = final[key]["ba"][test_domain]["n_optimal"]
                own_opt = final[key][test_domain][test_domain]["n_optimal"]
                print(f"checkpoint {checkpoint}: ba-on-{test_domain}={ba_geo:.2f}s({ba_opt}/30) vs "
                      f"{test_domain}-on-{test_domain}={own_geo:.2f}s({own_opt}/30) -- "
                      f"ratio={ba_geo/own_geo:.2f}x", flush=True)
    print("SWEEP_BA_VS_RR_WS_DONE", flush=True)
