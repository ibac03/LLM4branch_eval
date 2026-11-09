#!/bin/bash
# One-time setup on Ubuntu 24.04 x86_64. Everything stays inside this folder:
#   .conda/            Miniforge with the "ecole-fork" env (exact versions from conda-linux-64.lock)
#   LLM4Branch/        official repo (hzn18), fixed commit
#   LLM4Branch-ecole/  the authors' custom Ecole, fixed commit, built into the env
#
#   bash setup/install.sh
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONDA_DIR=$ROOT/.conda
LLM4BRANCH_COMMIT=e001d8c9f9f9b7d414922526c8d958a2cbf62462
ECOLE_COMMIT=debd0e44dc5c5c200fcb3845d511cd0729baaa3b
mkdir -p "$ROOT/tmp"

echo "== miniforge -> .conda/"
if [ ! -d "$CONDA_DIR" ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && rm /tmp/miniforge.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "== official repos"
clone_at() { [ -d "$2/.git" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; }
clone_at https://github.com/hzn18/LLM4Branch "$ROOT/LLM4Branch" $LLM4BRANCH_COMMIT
clone_at https://github.com/hzn18/LLM4Branch-ecole "$ROOT/LLM4Branch-ecole" $ECOLE_COMMIT

echo "== conda env ecole-fork"
conda env list | grep -q "^ecole-fork " || conda create -y -q -n ecole-fork --file "$ROOT/setup/conda-linux-64.lock"
conda activate ecole-fork
pip install -q -r "$ROOT/setup/pip-requirements.txt"
pip install -q --no-deps -e "$ROOT/LLM4Branch"   # provides the openevolve package

echo "== build the custom Ecole (a few minutes)"
cd "$ROOT/LLM4Branch-ecole"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_ARGS="-DSCIP_DIR=$CONDA_PREFIX/lib/cmake/scip -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_BUILD_TYPE=Release -DECOLE_BUILD_TESTS=OFF"
python setup.py develop > "$ROOT/tmp/ecole_build.log" 2>&1 || { tail -30 "$ROOT/tmp/ecole_build.log"; exit 1; }
python -c "import ecole; assert hasattr(ecole.observation, 'Khalil2016Flash'); print('ecole', ecole.__version__, 'OK')"
conda clean -a -y -q
echo "SETUP_DONE  (activate with: source .conda/etc/profile.d/conda.sh && conda activate ecole-fork)"
