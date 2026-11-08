import sys, json
sys.path.insert(0, '.')
from multiprocessing import Pool
import worst_decile_early_signal as es

NEW_SEEDS = list(range(800000400, 800001300))  # the 900 new seeds from extend2

if __name__ == "__main__":
    print(f"=== extending early-signal: {len(NEW_SEEDS)} more instances ===", flush=True)
    with Pool(processes=6) as pool:
        new_records = pool.map(es.early_signal, NEW_SEEDS)
    existing = json.load(open(es.RESULTS_PATH))
    existing_seeds = {r['seed'] for r in existing}
    new_records = [r for r in new_records if r['seed'] not in existing_seeds]
    combined = existing + new_records
    combined.sort(key=lambda r: r['seed'])
    json.dump(combined, open(es.RESULTS_PATH, 'w'), indent=2)
    print(f"merged: {len(existing)} + {len(new_records)} = {len(combined)} total", flush=True)
    print("=== EXTEND_EARLY_SIGNAL_COMPLETE ===", flush=True)
