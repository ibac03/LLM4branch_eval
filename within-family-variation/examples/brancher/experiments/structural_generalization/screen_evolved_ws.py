"""
Cheap screen: RPB vs Evolved-WS (native params) on small BA and RR samples.
Decides whether Evolved-WS needs a domain-specific retune before being
folded into the final oracle table's BA/RR columns, per the established
cheap-screen-then-retune-if-needed methodology.

BA screen seeds: first 4 of the paper's own BA test-seed range
(999999999..999999999+29), i.e. [999999999, 1000000000, 1000000001, 1000000002].
RR screen seeds: first 4 of the RR test-seed range (1000..1029), i.e. [1000,1001,1002,1003].
"""
import sys
import os
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import load_program, geometric_mean_stable, create_instance
from eval_category_b import solve_llm4branch, solve_rpb, TIME_LIMIT
from category_rr_indset import generate_instance as generate_rr_instance

SG = os.path.dirname(__file__)
WS_PROGRAM = os.path.join(SG, "ws_evolution_output", "best", "best_program.py")

BA_SEEDS = [999999999, 1000000000, 1000000001, 1000000002]
RR_SEEDS = [1000, 1001, 1002, 1003]


def solve_evolved_ws_on(args):
    domain, seed = args
    score_function, used_features, function_params, _ = load_program(WS_PROGRAM)
    score_function = partial(score_function, params=function_params)
    if domain == "BA":
        instance = create_instance("indset", seed=seed, level="medium")
    else:
        instance = generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    t, n, s = solve_llm4branch(instance, score_function, used_features)
    return t, n, s


def solve_rpb_on(args):
    domain, seed = args
    if domain == "BA":
        instance = create_instance("indset", seed=seed, level="medium")
    else:
        instance = generate_rr_instance(seed=seed, n_nodes=250, degree=3)
    t, n, s = solve_rpb(instance)
    return t, n, s


if __name__ == "__main__":
    print(f"TIME_LIMIT={TIME_LIMIT}", flush=True)

    for domain, seeds in [("BA", BA_SEEDS), ("RR", RR_SEEDS)]:
        print(f"=== {domain} screen ===", flush=True)
        with Pool(processes=4) as pool:
            rpb_results = pool.map(solve_rpb_on, [(domain, s) for s in seeds])
        rpb_times = [r[0] for r in rpb_results]
        rpb_geo = geometric_mean_stable(rpb_times)
        print(f"RPB: times={[round(t,2) for t in rpb_times]} statuses={[r[2] for r in rpb_results]} "
              f"geomean={rpb_geo:.2f}", flush=True)

        with Pool(processes=4) as pool:
            ws_results = pool.map(solve_evolved_ws_on, [(domain, s) for s in seeds])
        ws_times = [r[0] for r in ws_results]
        ws_statuses = [r[2] for r in ws_results]
        all_optimal = all(s == "optimal" for s in ws_statuses)
        ws_geo = geometric_mean_stable(ws_times)
        speedup = rpb_geo / ws_geo if ws_geo > 0 else 0
        print(f"Evolved-WS (native) on {domain}: times={[round(t,2) for t in ws_times]} "
              f"statuses={ws_statuses} all_optimal={all_optimal} geomean={ws_geo:.2f} "
              f"speedup_vs_RPB={speedup:.2f}x", flush=True)

    print("SCREEN_EVOLVED_WS_DONE", flush=True)
