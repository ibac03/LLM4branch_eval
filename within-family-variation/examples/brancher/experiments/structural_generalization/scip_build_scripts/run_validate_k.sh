#!/bin/bash
source ~/Desktop/scip-build/venv/bin/activate
export DYLD_LIBRARY_PATH=$HOME/Desktop/scip-build/install/lib:$DYLD_LIBRARY_PATH
cd /Users/hs/Desktop/LLM-4-Branch
export PYTHONPATH=$(pwd)
echo "=== Validate K tuning harness starting $(date) ==="
python ./examples/brancher/experiments/structural_generalization/validate_k_tuning_harness.py
echo "=== Validate K tuning harness finished $(date) ==="
