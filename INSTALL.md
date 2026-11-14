# Installing PCAccumulation on modern GPUs (RTX 30/40/50 series)

The README's setup (PyTorch 1.12 + CUDA 11.6) is from 2022. It does not support Ada (RTX 40xx,
sm_89) or newer GPUs, and the old pins conflict with modern Python, compilers, and numpy. This guide
covers a setup that works, and explains how to adapt it to a different machine.

Tested on: RTX 4060 Laptop (sm_89), NVIDIA driver 595 (CUDA 13.2), Ubuntu with
gcc 15 default + gcc-13, system nvcc 12.4, Python 3.10.

## TL;DR

```bash
bash setup_env.sh                       # no sudo needed
source venv_pcaccumulation/bin/activate
python dummy-test.py                    # sanity check without the dataset
```

## Final versions

| Component      | README             | Used here                   | Why |
|----------------|--------------------|-----------------------------|-----|
| Python         | 3.8 / 3.10         | 3.10                        | torch 2.4 wheels exist; old enough for the code |
| PyTorch        | 1.12.0+cu116       | 2.4.1+cu121                 | Native sm_89 kernels; torch-scatter wheels exist |
| torchsparse    | 1.4.0              | 1.4.0 (built from source)   | Code only uses `sparse_quantize`; 1.4 builds fine on torch 2.4 |
| torch-scatter  | for torch 1.12     | 2.1.2+pt24cu121             | Prebuilt wheel from data.pyg.org |
| numpy          | any                | **< 2** (1.26.4)            | numpy 2 breaks old compiled code and removed aliases |
| Host compiler  | gcc 9/11           | gcc-13                      | nvcc 12.4 rejects gcc > 13 |

## Step by step (what `setup_env.sh` does)

1. **Check what you have**
   ```bash
   nvidia-smi                       # driver's max CUDA version (top right) and GPU model
   nvidia-smi --query-gpu=compute_cap --format=csv   # e.g. 8.9 for RTX 40xx
   nvcc --version                   # system CUDA toolkit used to compile extensions
   ls /usr/bin | grep -E '^gcc-[0-9]+$'
   ```

2. **Pick a PyTorch build.** Rules:
   - Its CUDA version must support your GPU: sm_89 needs CUDA ≥ 11.8, sm_120 (RTX 50xx) needs ≥ 12.8.
   - Its CUDA **major** version must match your `nvcc` (12.x wheel with a 12.x nvcc). A minor
     mismatch only produces a warning.
   - It must not need a newer CUDA than your driver supports (shown in `nvidia-smi`).
   - torch-scatter must publish wheels for it: check `https://data.pyg.org/whl/`.

3. **Pick a host compiler that your nvcc accepts.** Ubuntu's default gcc is often too new.
   Maximum supported gcc: CUDA 11.8 → gcc 11, 12.0–12.3 → gcc 12, 12.4–12.8 → gcc 13.
   `sudo apt install gcc-13 g++-13` if needed. Point the build at it with
   `CC`, `CXX`, `CUDAHOSTCXX` and `NVCC_PREPEND_FLAGS="-ccbin g++-13"`.

4. **Get the sparsehash headers** (needed by torchsparse 1.4). Normally
   `sudo apt install libsparsehash-dev`. Without sudo (e.g. no terminal for the password), unpack
   the package locally, since it is header-only:
   ```bash
   mkdir -p .deps/deb && cd .deps/deb && apt-get download libsparsehash-dev
   dpkg -x libsparsehash-dev_*.deb ../sparsehash && cd ../..
   export CPLUS_INCLUDE_PATH=$PWD/.deps/sparsehash/usr/include
   ```

5. **Create the venv and install wheels**
   ```bash
   python3.10 -m venv venv_pcaccumulation
   venv_pcaccumulation/bin/pip install --upgrade pip "setuptools<70" wheel ninja
   venv_pcaccumulation/bin/pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
       --index-url https://download.pytorch.org/whl/cu121
   venv_pcaccumulation/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   venv_pcaccumulation/bin/pip install "numpy<2" scipy scikit-learn pandas numba tqdm pyyaml \
       tensorboardX filterpy nestargs ipython open3d termcolor pyquaternion matplotlib
   ```
   `ninja` is required: `chamfer_distance/` is JIT-compiled via `torch.utils.cpp_extension.load`.

6. **Persist the build variables** in `venv_pcaccumulation/bin/activate`. The chamfer
   extension compiles itself the first time it's imported, so these variables are still needed
   after setup:
   ```bash
   export CUDA_HOME=/usr                      # parent of bin/nvcc
   export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13
   export NVCC_PREPEND_FLAGS="-ccbin g++-13"
   export TORCH_CUDA_ARCH_LIST="8.9"          # your compute capability; keeps builds fast
   export CPLUS_INCLUDE_PATH=<sparsehash include dir>
   ```

7. **Build torchsparse** (after `source venv_pcaccumulation/bin/activate`):
   ```bash
   MAX_JOBS=4 pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
   ```
   `--no-build-isolation` lets the build see the installed torch. `MAX_JOBS` limits parallel nvcc
   processes. Each one can use several GB of RAM, so lower it on machines with less memory.

8. **Verify** with `python dummy-test.py`. It loads the checkpoint onto the GPU and builds a
   torchsparse tensor. A stronger check is to build `MotionNet` from
   `configs/waymo/waymo.yaml` and call `load_state_dict(..., strict=True)` with
   `checkpoints/waymo.pth`. All keys should match.

## Code changes needed for newer torch/numpy

These are already committed on this branch:
- `libs/trainer.py`, `libs/tester.py`: `c_loader_iter.next()` → `next(c_loader_iter)`
  (newer torch DataLoader iterators no longer have `.next()`).
- `libs/tester.py`, `toolbox/sf_eval_utils.py`, `toolbox/cluster_eval.py`: `np.bool`/`np.float`
  → `bool`/`float` (removed in numpy 1.24).

To find similar breakage in other old repos:
```bash
grep -rnE "\.next\(\)|np\.(float|int|bool|object|long)\b[^_0-9]" --include=*.py .
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unsupported GNU version! gcc versions later than 13 are not supported` | Use an older gcc (step 3) |
| `google/dense_hash_map: No such file or directory` | sparsehash headers missing (step 4) |
| `Ninja is required to load C++ extensions` | `pip install ninja` |
| `no kernel image is available for execution on the device` | torch/extension built for the wrong arch: check `TORCH_CUDA_ARCH_LIST` and the torch CUDA version |
| `A module compiled using NumPy 1.x cannot be run in NumPy 2` | `pip install "numpy<2"` |
| `module 'numpy' has no attribute 'bool'/'float'` | Replace with builtins (see above) |
| `'_MultiProcessingDataLoaderIter' object has no attribute 'next'` | Use `next(it)` |
| `sudo: A terminal is required to authenticate` | Run sudo in your own terminal, or use the local unpack in step 4 |
| CUDA out of memory on 8 GB cards | Lower the batch size: `python main.py configs/waymo/waymo.yaml 2 1 ...` |

## Not covered

`dataset_toolbox/` (raw data preprocessing) additionally needs tensorflow, waymo-open-dataset
and nuscenes-devkit. You don't need them if you use the preprocessed `data.zip`.
