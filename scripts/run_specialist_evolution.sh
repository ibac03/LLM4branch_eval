#!/bin/bash
# Task 4: launch the specialist evolution (Hardik's openevolve setup, easy instances) in the background.
# Needs within-family-variation/configs/my_key_set.yaml: copy my_key_set.example.yaml and add the API key.
# Settings: within-family-variation/examples/brancher/config_specialist_easy.yaml and
#           .../structural_generalization/run_specialist_easy.py (CORES=<range> to override, e.g. 0-63)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/within-family-variation
SG=$W/examples/brancher/experiments/structural_generalization
[ -f "$W/configs/my_key_set.yaml" ] || { echo "missing $W/configs/my_key_set.yaml"; exit 1; }
mkdir -p "$ROOT/results/runs" "$W/examples/brancher/tmp/cpu_locks/specialist_evolution"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ecole-fork
cd "$W"   # config paths (./configs/...) are relative to this folder
# $W first on PYTHONPATH so Hardik's openevolve (with the OpenCode changes) wins over the official one
PYTHONPATH="$W:$W/examples/brancher:$SG" nohup setsid python "$SG/run_specialist_easy.py" \
  > "$ROOT/results/runs/evo.log" 2>&1 < /dev/null &
echo "started (pid $!); log: results/runs/evo.log"
