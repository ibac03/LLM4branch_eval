"""
Final 4-row oracle table (Original, Evolved-RR, Evolved-WS, Oracle) over
the combined 90-instance BA+RR+WS test set (30 same-seeded instances per
domain, per row). Parses raw per-instance times directly from the
existing run logs (no re-running), pools them per row, and computes
geometric_mean_stable() directly on the pooled set (no geomean-of-geomeans
shortcut), per the established oracle-table methodology.
"""
import os
import re
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "log", "experiments")


def geometric_mean_stable(data, shift=1):
    a = np.asarray(data)
    if a.size == 0 or np.any(a < 0):
        return 0.0
    return float(np.exp(np.mean(np.log(a + shift)))) - shift


def parse_ba_paper_log(path, seed_base=999999999, n=30, start_marker=None, end_marker=None):
    """Parses 'Core: X, Seed: S, Method: [...], Proc Time: [T], Node: [N]' lines.

    start_marker/end_marker bound the section to read from, since some logs
    (e.g. ba_paper_seeds.log) contain two separate runs reusing the exact
    same seed range -- without bounding, a naive whole-file scan silently
    picks up whichever run's line for a given seed appears LAST in the
    file, corrupting the earlier run's numbers with the later run's.
    """
    text = open(path).read()
    if start_marker is not None:
        text = text[text.index(start_marker):]
    if end_marker is not None:
        text = text[:text.index(end_marker)]
    pattern = re.compile(r"Seed: (\d+), Method: \[.*?\], Proc Time: \[([\d.]+)\]")
    times_by_seed = {}
    for seed_str, t_str in pattern.findall(text):
        seed = int(seed_str)
        if seed_base <= seed < seed_base + n:
            times_by_seed[seed] = float(t_str)
    seeds = list(range(seed_base, seed_base + n))
    missing = [s for s in seeds if s not in times_by_seed]
    if missing:
        raise ValueError(f"{path}: missing seeds {missing}")
    return [times_by_seed[s] for s in seeds]


def parse_seedtime_log(path, label_prefix, seeds):
    """Parses 'seed=S: LABEL(time=T, nodes=N, status=...)' lines."""
    text = open(path).read()
    pattern = re.compile(re.escape(label_prefix) + r"\(time=([\d.]+)")
    seed_pattern = re.compile(r"seed=(\d+): " + re.escape(label_prefix) + r"\(time=([\d.]+)")
    times_by_seed = {int(s): float(t) for s, t in seed_pattern.findall(text)}
    missing = [s for s in seeds if s not in times_by_seed]
    if missing:
        raise ValueError(f"{path} [{label_prefix}]: missing seeds {missing}")
    return [times_by_seed[s] for s in seeds]


BA_SEEDS = list(range(999999999, 999999999 + 30))
RR_SEEDS = list(range(1000, 1030))
WS_SEEDS = list(range(200, 230))

# --- Original ---
original_ba = parse_ba_paper_log(
    os.path.join(LOG_DIR, "ba_paper_seeds.log"), seed_base=999999999, n=30,
    start_marker="=== Original on BA-standard", end_marker="=== Original on BA-standard done")
original_rr = parse_seedtime_log(os.path.join(LOG_DIR, "rr_n250_rerun.log"), "Original", RR_SEEDS)
original_ws = parse_seedtime_log(os.path.join(LOG_DIR, "eval_ws_testset.log"), "Original", WS_SEEDS)

# --- Evolved-RR (BA: tuned, RR: native, WS: native) ---
evolved_rr_ba = parse_ba_paper_log(
    os.path.join(LOG_DIR, "evolved_rr_ba_tuned_crosstest.log"), seed_base=999999999, n=30,
    start_marker="=== Evolved-RR-tuned-on-BA (BA-standard", end_marker="=== Evolved-RR-tuned-on-BA (BA-standard) done")
evolved_rr_rr = parse_seedtime_log(os.path.join(LOG_DIR, "evolved_rr_crosstest.log"), "Evolved-RR-on-RR", RR_SEEDS)
evolved_rr_ws = parse_seedtime_log(os.path.join(LOG_DIR, "eval_ws_testset.log"), "Evolved-RR", WS_SEEDS)

# --- Evolved-WS (BA: tuned, RR: native, WS: native) ---
evolved_ws_ba = parse_ba_paper_log(
    os.path.join(LOG_DIR, "eval_evolved_ws_ba_tuned.log"), seed_base=999999999, n=30,
    start_marker="=== Evolved-WS-tuned-on-BA on BA-standard", end_marker="=== Evolved-WS-tuned-on-BA on BA-standard done")
evolved_ws_rr = parse_seedtime_log(os.path.join(LOG_DIR, "eval_evolved_ws_rr.log"), "Evolved-WS-on-RR", RR_SEEDS)
evolved_ws_ws = parse_seedtime_log(os.path.join(LOG_DIR, "eval_ws_testset.log"), "Evolved-WS", WS_SEEDS)

rows = {
    "Original": original_ba + original_rr + original_ws,
    "Evolved-RR": evolved_rr_ba + evolved_rr_rr + evolved_rr_ws,
    "Evolved-WS": evolved_ws_ba + evolved_ws_rr + evolved_ws_ws,
}

for label, times in rows.items():
    assert len(times) == 90, f"{label}: expected 90, got {len(times)}"

oracle_times = [min(rows["Original"][i], rows["Evolved-RR"][i], rows["Evolved-WS"][i]) for i in range(90)]

print("=== Per-domain geomeans (sanity check against known values) ===")
for label, (ba, rr, ws) in [
    ("Original", (original_ba, original_rr, original_ws)),
    ("Evolved-RR", (evolved_rr_ba, evolved_rr_rr, evolved_rr_ws)),
    ("Evolved-WS", (evolved_ws_ba, evolved_ws_rr, evolved_ws_ws)),
]:
    print(f"{label}: BA={geometric_mean_stable(ba):.2f} RR={geometric_mean_stable(rr):.2f} WS={geometric_mean_stable(ws):.2f}")

print("\n=== Final pooled 4-row oracle table (90 combined BA+RR+WS instances) ===")
for label, times in rows.items():
    print(f"{label}: pooled_geomean={geometric_mean_stable(times):.2f}")
oracle_geo = geometric_mean_stable(oracle_times)
print(f"Oracle: pooled_geomean={oracle_geo:.2f}")

best_fixed = min(geometric_mean_stable(times) for times in rows.values())
print(f"\nBest single fixed policy geomean: {best_fixed:.2f}")
print(f"Oracle geomean: {oracle_geo:.2f}")
print(f"Oracle-gap (best_fixed / oracle): {best_fixed / oracle_geo:.2f}x")

# Win counts (which row wins each instance)
win_counts = {label: 0 for label in rows}
for i in range(90):
    best_label = min(rows, key=lambda l: rows[l][i])
    win_counts[best_label] += 1
print(f"\nWin counts across 90 instances: {win_counts}")
