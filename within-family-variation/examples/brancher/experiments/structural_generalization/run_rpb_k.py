"""
Run RPB (SCIP's reliability pseudocost branching) on both Category K (ER,
n=400, same 30 seeds as the crosstest) and standard IndSet (BA affinity=4,
n=1000, same 30 seeds as the crosstest), for comparison against Original
and Tuned-K.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import geometric_mean_stable, create_instance
from eval_category_b import solve_rpb
from category_k_er_indset import generate_instance as generate_er_instance

print("=== RPB on Category K (ER, n=400), seeds 100-129 ===", flush=True)
times = []
for seed in range(100, 130):
    instance = generate_er_instance(seed=seed)
    t, n, s = solve_rpb(instance)
    times.append(t)
    print(f"seed={seed}: RPB(time={t:.2f}, nodes={n}, status={s})", flush=True)
print(f"RPB-on-CategoryK geomean: {geometric_mean_stable(times):.2f}", flush=True)

print("=== RPB on standard IndSet (BA affinity=4, n=1000), seeds 2000000000-2000000029 ===", flush=True)
times = []
for seed in range(2000000000, 2000000030):
    instance = create_instance('indset', seed, 'medium')
    t, n, s = solve_rpb(instance)
    times.append(t)
    print(f"seed={seed}: RPB(time={t:.2f}, nodes={n}, status={s})", flush=True)
print(f"RPB-on-standard geomean: {geometric_mean_stable(times):.2f}", flush=True)
