"""
Extend worst_decile_analysis.py's 100 instances to 400, for more statistical
power on the weak-but-real correlates found so far (root_gap's Pearson
r=0.32 vs ratio, clique_size_cv's r=-0.25 vs ratio -- both currently only
significant under the outlier-sensitive test, not the robust one). Reuses
run_one_seed exactly as-is (same methodology) for the 300 NEW seeds
(800000100-800000399), then merges with the existing 100
(800000000-800000099) into one 400-instance dataset.
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import worst_decile_analysis as base

NEW_SEEDS = list(range(800000100, 800000400))  # 300 new -> 400 total with the original 100

if __name__ == "__main__":
    print(f"=== extending worst-decile analysis: {len(NEW_SEEDS)} more BA instances "
          f"(seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]}) ===", flush=True)
    with Pool(processes=base.WORKERS) as pool:
        new_records = pool.map(base.run_one_seed, NEW_SEEDS)

    existing = json.load(open(base.RESULTS_PATH))
    existing_seeds = {r["seed"] for r in existing}
    new_records = [r for r in new_records if r["seed"] not in existing_seeds]  # safety: no dupes
    combined = existing + new_records
    combined.sort(key=lambda r: r["seed"])
    json.dump(combined, open(base.RESULTS_PATH, "w"), indent=2)

    print(f"\nmerged: {len(existing)} existing + {len(new_records)} new = {len(combined)} total", flush=True)
    print(f"wrote {base.RESULTS_PATH}", flush=True)
    print("\n=== WORST_DECILE_EXTEND_COMPLETE ===", flush=True)
