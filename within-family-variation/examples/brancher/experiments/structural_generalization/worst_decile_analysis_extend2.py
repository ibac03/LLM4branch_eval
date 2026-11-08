"""
Second extension: 400 -> 1300 instances. ~873 instances matches ~10h of
compute at the blended observed rate (0.688 min/instance across the first
two batches: 100 in 41min, 300 in 234min); rounded to 900 for a clean
number. Same run_one_seed function, new seed range (800000400-800001299),
merged into the same results file.
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import worst_decile_analysis as base

NEW_SEEDS = list(range(800000400, 800001300))  # 900 new -> 1300 total

if __name__ == "__main__":
    print(f"=== extending worst-decile analysis: {len(NEW_SEEDS)} more BA instances "
          f"(seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]}) ===", flush=True)
    with Pool(processes=base.WORKERS) as pool:
        new_records = pool.map(base.run_one_seed, NEW_SEEDS)

    existing = json.load(open(base.RESULTS_PATH))
    existing_seeds = {r["seed"] for r in existing}
    new_records = [r for r in new_records if r["seed"] not in existing_seeds]
    combined = existing + new_records
    combined.sort(key=lambda r: r["seed"])
    json.dump(combined, open(base.RESULTS_PATH, "w"), indent=2)

    print(f"\nmerged: {len(existing)} existing + {len(new_records)} new = {len(combined)} total", flush=True)
    print(f"wrote {base.RESULTS_PATH}", flush=True)
    print("\n=== WORST_DECILE_EXTEND2_COMPLETE ===", flush=True)
