"""Cheap early-search probe for the policy selector, on the official 1300 easy test seeds.

Same probe as the PDF author's worst_decile_early_signal.py (run the original LLM4Branch policy for
the first K B&B nodes through the custom-Ecole Branching env and read the live gap), at n=750 and
with extra checkpoints. Run inside the ecole-fork env (CORES=<n> to limit the cores used):

    python scripts/probe_selector_features.py external/LLM4Branch results/runs/probe_easy.json
"""
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

REPO = os.path.expanduser(sys.argv[1])
OUT = os.path.expanduser(sys.argv[2])
sys.path.insert(0, os.path.join(REPO, "examples", "brancher"))
from utils.utils import load_program, get_scip_params, normalize  # noqa: E402

PROGRAM = os.path.join(REPO, "examples", "brancher", "program", "indset", "program.py")
SEEDS = list(range(999999999, 999999999 + 1300))
CHECKPOINTS = [1, 10, 25, 50]
SAFETY_TIME_LIMIT = 120


def probe(seed):
    import ecole
    score_function, used_features, params, _ = load_program(PROGRAM)
    score_function = partial(score_function, params=params)
    gen = ecole.instance.IndependentSetGenerator(
        n_nodes=750, affinity=4, graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    gen.seed(seed)
    instance = next(gen)
    khalil, node = ecole.observation.Khalil2016Flash(), ecole.observation.NodeBipartiteFlash()
    khalil.set_active_features([f - 19 for f in used_features if f >= 19])
    node.set_active_features([f for f in used_features if f < 19])
    env = ecole.environment.Branching(observation_function={"Node": node, "Khalil": khalil},
                                      scip_params=get_scip_params(time_limit=SAFETY_TIME_LIMIT))
    env.seed(0)
    t0 = time.process_time()
    obs, action_set, _, done, _ = env.reset(instance)
    m = env.model.as_pyscipopt()
    rec = {"seed": seed, "root_time": time.process_time() - t0, "root_dual": m.getDualbound(),
           "root_primal": m.getPrimalbound(), "root_ncands": 0 if done else len(action_set)}
    todo = list(CHECKPOINTS)
    while not done and todo:
        n = m.getNNodes()
        while todo and n >= todo[0]:
            k = todo.pop(0)
            rec[f"gap_{k}"] = m.getGap()
            rec[f"time_{k}"] = time.process_time() - t0
        if not todo:
            break
        f = np.concatenate((obs["Node"].variable_features[action_set, :], obs["Khalil"].features[action_set, :]), axis=1)
        f = normalize(f, used_features)
        obs, action_set, _, done, _ = env.step(action_set[score_function(f).argmax()])
        m = env.model.as_pyscipopt()
    for k in todo:  # solved before reaching this checkpoint
        rec[f"gap_{k}"] = 0.0 if done else m.getGap()
        rec[f"time_{k}"] = time.process_time() - t0
    rec["finished_early"] = bool(done)
    rec["nodes_reached"] = m.getNNodes()
    return rec


if __name__ == "__main__":
    t = time.time()
    with Pool(int(os.environ.get("CORES", os.cpu_count()))) as pool:
        recs = pool.map(probe, SEEDS, chunksize=4)
    json.dump(sorted(recs, key=lambda r: r["seed"]), open(OUT, "w"), indent=1)
    print(f"probed {len(recs)} instances in {(time.time() - t) / 60:.1f} min -> {OUT}")
