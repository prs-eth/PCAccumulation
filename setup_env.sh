#!/usr/bin/env bash
# Reproducible environment setup for PCAccumulation on modern NVIDIA GPUs
# (tested: RTX 4060 Laptop / sm_89, driver 595, Ubuntu, system nvcc 12.4, gcc-13).
# No sudo required. See INSTALL.md for the reasoning behind each step.
#
# Usage:   bash setup_env.sh
# Override defaults with env vars, e.g.:
#   VENV=myvenv HOST_GCC=12 CUDA_ARCH=8.6 bash setup_env.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-venv_pcaccumulation}"
PYTHON="${PYTHON:-python3.10}"
HOST_GCC="${HOST_GCC:-13}"                 # must be supported by your nvcc (nvcc 12.4 -> gcc <= 13)
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}"
TORCH_VER="${TORCH_VER:-2.4.1}"
TORCHVISION_VER="${TORCHVISION_VER:-0.19.1}"
TORCHAUDIO_VER="${TORCHAUDIO_VER:-2.4.1}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"          # must have the same CUDA *major* version as your nvcc
SCATTER_INDEX="${SCATTER_INDEX:-https://data.pyg.org/whl/torch-2.4.0+${TORCH_CUDA}.html}"

cd "$REPO"
echo ">> repo=$REPO venv=$VENV python=$PYTHON gcc-$HOST_GCC CUDA_HOME=$CUDA_HOME arch=$CUDA_ARCH torch=$TORCH_VER+$TORCH_CUDA"

command -v "gcc-$HOST_GCC" >/dev/null || { echo "gcc-$HOST_GCC not found (sudo apt install gcc-$HOST_GCC g++-$HOST_GCC)"; exit 1; }
command -v "$PYTHON" >/dev/null       || { echo "$PYTHON not found"; exit 1; }

# 1. Google sparsehash headers (needed to compile torchsparse 1.4). Header-only: unpack the
#    Ubuntu package locally instead of installing it system-wide.
SPARSEHASH_INC=/usr/include
if [ ! -e /usr/include/google/dense_hash_map ]; then
  SPARSEHASH_INC="$REPO/.deps/sparsehash/usr/include"
  if [ ! -e "$SPARSEHASH_INC/google/dense_hash_map" ]; then
    mkdir -p "$REPO/.deps/deb"
    (cd "$REPO/.deps/deb" && apt-get download libsparsehash-dev)
    dpkg -x "$REPO"/.deps/deb/libsparsehash-dev_*.deb "$REPO/.deps/sparsehash"
  fi
fi

# 2. Virtual environment
[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
PIP="$REPO/$VENV/bin/pip"
"$PIP" install --upgrade pip "setuptools<70" wheel ninja

# 3. PyTorch + torch-scatter (prebuilt wheels)
"$PIP" install "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" "torchaudio==$TORCHAUDIO_VER" \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
"$PIP" install torch-scatter -f "$SCATTER_INDEX"

# 4. Python dependencies used by main.py / libs / models / toolbox
"$PIP" install "numpy<2" scipy scikit-learn pandas numba tqdm pyyaml tensorboardX \
  filterpy nestargs ipython open3d termcolor pyquaternion matplotlib \
  "onnx==1.17.0" "onnxruntime==1.20.1"

# 5. Build env vars, persisted in the venv's activate script (needed again later, because the
#    chamfer_distance CUDA extension is JIT-compiled the first time it is imported).
MARK="# --- PCAccumulation build settings ---"
if ! grep -qF "$MARK" "$VENV/bin/activate"; then
  cat >> "$VENV/bin/activate" <<EOF

$MARK
export CUDA_HOME="$CUDA_HOME"
export CC=gcc-$HOST_GCC CXX=g++-$HOST_GCC CUDAHOSTCXX=g++-$HOST_GCC
export NVCC_PREPEND_FLAGS="-ccbin g++-$HOST_GCC"
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export CPLUS_INCLUDE_PATH="$SPARSEHASH_INC\${CPLUS_INCLUDE_PATH:+:\$CPLUS_INCLUDE_PATH}"
EOF
fi

# 6. torchsparse 1.4.0 from source (MAX_JOBS limits parallel nvcc processes / RAM use)
# shellcheck disable=SC1090
source "$VENV/bin/activate"
MAX_JOBS="${MAX_JOBS:-4}" pip install --no-build-isolation \
  git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# 7. Smoke test
python dummy-test.py
echo ">> Done. Activate with: source $VENV/bin/activate"
