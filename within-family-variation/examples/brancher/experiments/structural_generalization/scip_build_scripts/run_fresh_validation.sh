#!/bin/bash
source ~/Desktop/scip-build/venv/bin/activate
export DYLD_LIBRARY_PATH=$HOME/Desktop/scip-build/install/lib:$DYLD_LIBRARY_PATH
cd /Users/hs/Desktop/LLM-4-Branch
export PYTHONPATH=$(pwd)
set -e

SG=./examples/brancher/experiments/structural_generalization

echo "=== FRESH VALIDATION: 150 standard SetCover + 150 Category G, 6 arms ==="

# --- Standard SetCover, one arm at a time ---
echo "--- [1/12] Standard SetCover: RPB starting $(date) ---"
python $SG/eval_arm_standard_fresh.py --arm_name RPB --method relpscost
echo "--- [1/12] done $(date) ---"

echo "--- [2/12] Standard SetCover: Original starting $(date) ---"
python $SG/eval_arm_standard_fresh.py --arm_name Original --method evolve \
    --program_path ./examples/brancher/program/setcover/program.py
echo "--- [2/12] done $(date) ---"

echo "--- [3/12] Standard SetCover: Tuned-123 starting $(date) ---"
python $SG/eval_arm_standard_fresh.py --arm_name Tuned-123 --method evolve \
    --program_path $SG/program_category_g_tuned.py
echo "--- [3/12] done $(date) ---"

echo "--- [4/12] Standard SetCover: Tuned-1 starting $(date) ---"
python $SG/eval_arm_standard_fresh.py --arm_name Tuned-1 --method evolve \
    --program_path $SG/program_category_g_tuned_seed1.py
echo "--- [4/12] done $(date) ---"

echo "--- [5/12] Standard SetCover: Tuned-7 starting $(date) ---"
python $SG/eval_arm_standard_fresh.py --arm_name Tuned-7 --method evolve \
    --program_path $SG/program_category_g_tuned_seed7.py
echo "--- [5/12] done $(date) ---"

echo "--- [6/12] Standard SetCover: Tuned-42 starting $(date) ---"
python $SG/eval_arm_standard_fresh.py --arm_name Tuned-42 --method evolve \
    --program_path $SG/program_category_g_tuned_seed42.py
echo "--- [6/12] done $(date) ---"

# --- Category G, one arm at a time ---
echo "--- [7/12] Category G: RPB starting $(date) ---"
python $SG/eval_arm_categoryg_fresh.py --arm_name RPB
echo "--- [7/12] done $(date) ---"

echo "--- [8/12] Category G: Original starting $(date) ---"
python $SG/eval_arm_categoryg_fresh.py --arm_name Original \
    --program_path ./examples/brancher/program/setcover/program.py
echo "--- [8/12] done $(date) ---"

echo "--- [9/12] Category G: Tuned-123 starting $(date) ---"
python $SG/eval_arm_categoryg_fresh.py --arm_name Tuned-123 \
    --program_path $SG/program_category_g_tuned.py
echo "--- [9/12] done $(date) ---"

echo "--- [10/12] Category G: Tuned-1 starting $(date) ---"
python $SG/eval_arm_categoryg_fresh.py --arm_name Tuned-1 \
    --program_path $SG/program_category_g_tuned_seed1.py
echo "--- [10/12] done $(date) ---"

echo "--- [11/12] Category G: Tuned-7 starting $(date) ---"
python $SG/eval_arm_categoryg_fresh.py --arm_name Tuned-7 \
    --program_path $SG/program_category_g_tuned_seed7.py
echo "--- [11/12] done $(date) ---"

echo "--- [12/12] Category G: Tuned-42 starting $(date) ---"
python $SG/eval_arm_categoryg_fresh.py --arm_name Tuned-42 \
    --program_path $SG/program_category_g_tuned_seed42.py
echo "--- [12/12] done $(date) ---"

echo "ALL_FRESH_VALIDATION_DONE $(date)"
