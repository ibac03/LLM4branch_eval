#!/bin/bash
# One-time setup on a fresh Ubuntu 24.04 x86_64 machine:
#   - Miniforge in ~/miniforge3 (skipped if present)
#   - the official LLM4Branch repo and its custom Ecole fork, cloned into external/ at the exact commits used
#   - the "ecole-fork" conda env with the exact package versions of the EC2 runs, then builds the Ecole fork
#
#   bash setup/install.sh                 # ENV_NAME=other-name bash setup/install.sh for a second env
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_NAME=${ENV_NAME:-ecole-fork}
LLM4BRANCH_COMMIT=e001d8c9f9f9b7d414922526c8d958a2cbf62462
ECOLE_COMMIT=debd0e44dc5c5c200fcb3845d511cd0729baaa3b
mkdir -p "$ROOT/external" "$ROOT/tmp"

echo "== miniforge"
if [ ! -d ~/miniforge3 ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p ~/miniforge3 && rm /tmp/miniforge.sh
fi
source ~/miniforge3/etc/profile.d/conda.sh

echo "== official repos -> external/"
clone_at() { [ -d "$2/.git" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; }
clone_at https://github.com/hzn18/LLM4Branch "$ROOT/external/LLM4Branch" $LLM4BRANCH_COMMIT
clone_at https://github.com/hzn18/LLM4Branch-ecole "$ROOT/external/LLM4Branch-ecole" $ECOLE_COMMIT

echo "== conda env $ENV_NAME"
conda env list | grep -q "^$ENV_NAME " || conda create -y -q -n "$ENV_NAME" --file "$ROOT/setup/conda-linux-64.lock"
conda activate "$ENV_NAME"
pip install -q -r "$ROOT/setup/pip-requirements.txt"
pip install -q --no-deps -e "$ROOT/external/LLM4Branch"   # provides the openevolve package

echo "== build the Ecole fork (a few minutes)"
cd "$ROOT/external/LLM4Branch-ecole"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_ARGS="-DSCIP_DIR=$CONDA_PREFIX/lib/cmake/scip -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_BUILD_TYPE=Release -DECOLE_BUILD_TESTS=OFF"
python setup.py develop > "$ROOT/tmp/ecole_build.log" 2>&1 || { tail -30 "$ROOT/tmp/ecole_build.log"; exit 1; }
python -c "import ecole; assert hasattr(ecole.observation, 'Khalil2016Flash'); print('ecole', ecole.__version__, 'OK')"
conda clean -a -y -q
echo "SETUP_DONE  (activate with: conda activate $ENV_NAME)"
