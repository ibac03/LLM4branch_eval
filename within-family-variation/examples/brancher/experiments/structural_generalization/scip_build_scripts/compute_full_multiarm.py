import csv
import re
import numpy as np

def gm(vals, shift=1.0):
    a = np.asarray(vals) + shift
    return np.exp(np.mean(np.log(a))) - shift

def load_csv(path, seed_col="seed", time_col="proc_time"):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[int(row[seed_col])] = float(row[time_col])
    return d

SC = "/Users/hs/Desktop/LLM-4-Branch/examples/brancher/experiments/structural_generalization/standard_comparison/"
RES = "/Users/hs/Desktop/LLM-4-Branch/examples/brancher/result/setcover/medium/relpscost.csv"

# --- Standard SetCover ---
std = {
    "RPB": load_csv(RES),
    "Original": load_csv(SC + "original_medium.csv"),
    "Tuned-123": load_csv(SC + "tuned_medium.csv"),
    "Tuned-1": load_csv(SC + "tuned_seed1_medium.csv"),
    "Tuned-7": load_csv(SC + "tuned_seed7_medium.csv"),
    "Tuned-42": load_csv(SC + "tuned_seed42_medium.csv"),
}
std_common = set.intersection(*[set(d) for d in std.values()])
std_pool = [{name: std[name][s] for name in std} for s in std_common]

# --- Category G ---
pat_rpb_llm = re.compile(
    r"seed=(\d+): RPB\(time=([\d.]+), nodes=(\d+), status=(\w+)\) "
    r"LLM4Branch\(time=([\d.]+), nodes=(\d+), status=(\w+)\)"
)
pat_tuned = re.compile(r"seed=(\d+): Tuned\(time=([\d.]+), nodes=(\d+), status=(\w+)\)")

LOG = "/Users/hs/Desktop/LLM-4-Branch/examples/brancher/log/experiments/"
catg_data = {}
for fpath in [LOG + "g08_natural_medium_300.log", LOG + "g08_natural_medium_340.log"]:
    for line in open(fpath):
        m = pat_rpb_llm.search(line)
        if m:
            seed, rpb_t, _, _, llm_t, _, _ = m.groups()
            catg_data.setdefault(int(seed), {})["RPB"] = float(rpb_t)
            catg_data[int(seed)]["Original"] = float(llm_t)

seed_files = {
    "Tuned-123": ["tuned_natural_300.log", "tuned_natural_340.log"],
    "Tuned-1": ["catg_tuned_seed1_300.log", "catg_tuned_seed1_340.log"],
    "Tuned-7": ["catg_tuned_seed7_300.log", "catg_tuned_seed7_340.log"],
    "Tuned-42": ["catg_tuned_seed42_300.log", "catg_tuned_seed42_340.log"],
}
for name, files in seed_files.items():
    for fname in files:
        for line in open(LOG + fname):
            m = pat_tuned.search(line)
            if m:
                seed, t, _, _ = m.groups()
                catg_data.setdefault(int(seed), {})[name] = float(t)

arms = ["RPB", "Original", "Tuned-123", "Tuned-1", "Tuned-7", "Tuned-42"]
catg_pool = [v for v in catg_data.values() if all(k in v for k in arms)]

print(f"Standard SetCover instances (all 6 arms): {len(std_pool)}")
print(f"Category G instances (all 6 arms): {len(catg_pool)}\n")

mixed_pool = std_pool + catg_pool

def report(label, pool):
    print(f"=== {label} (n={len(pool)}) ===")
    geomeans = {}
    for arm in arms:
        vals = [x[arm] for x in pool]
        geomeans[arm] = gm(vals)
    for arm in sorted(geomeans, key=geomeans.get):
        print(f"  {arm:12s}: {geomeans[arm]:6.2f}s")

    wins = {arm: 0 for arm in arms}
    for x in pool:
        best_arm = min(arms, key=lambda a: x[a])
        wins[best_arm] += 1
    print("  Wins:", {a: wins[a] for a in arms})

    oracle = [min(x[a] for a in arms) for x in pool]
    best_fixed_name = min(geomeans, key=geomeans.get)
    best_fixed_val = geomeans[best_fixed_name]
    oracle_gm = gm(oracle)
    gap = 100 * (best_fixed_val - oracle_gm) / best_fixed_val
    print(f"  Best fixed: {best_fixed_name} ({best_fixed_val:.2f}s)")
    print(f"  6-arm oracle: {oracle_gm:.2f}s")
    print(f"  Oracle gap vs best fixed: {gap:.1f}%\n")

report("Standard SetCover only", std_pool)
report("Category G only", catg_pool)
report("MIXED heterogeneous stream", mixed_pool)
