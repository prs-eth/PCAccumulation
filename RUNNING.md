# Running PCAccumulation

How to test, benchmark and regression-check the pretrained model on the selected real frames
(`data/waymo/test/`, list in `data/waymo/selected_frames.txt`).
To set up the environment first, see [INSTALL.md](INSTALL.md).

## 0. Activate the environment

Do this once in every new terminal:

```bash
cd ~/Desktop/PCAccumulation-SSC
source venv_pcaccumulation/bin/activate
```

## 1. Quick check that everything works (a few seconds)

```bash
python dummy-test.py
```

You should see two ✅ lines: the checkpoint loads onto the GPU, and torchsparse works.

## 2. Benchmark the speed (V0 baseline)

```bash
python benchmarks/bench_v0.py
```

Runs the original model (`test` mode, batch 1) on every selected frame: 10 warm-up frames, then 3
timed passes. Frames are read from disk outside the timed region. Each frame is timed per stage,
with the same stage boundaries the C++ pipeline will use:

| stage | what V0 does there |
|---|---|
| `preprocess` | crop + ground removal (CPU) |
| `voxelize` | numba 4D voxelisation + collate (CPU) |
| `h2d` | host → GPU copy |
| `pillar_enc` | pillar encoder + BEV scatter |
| `backbone` | UNet + fg/bg head + ego feature head |
| `ego_motion` | ego-motion head |
| `motion_seg` | warp + STPN |
| `cluster` | DBSCAN |
| `tubenet` | AlignNet (outside the C++ scope, not part of e2e) |

`e2e` is the sum of `preprocess` … `cluster`. Each stage waits for the GPU at its end, so the
stages add up. `e2e_nosync` is a second run of the same frame without those waits: the real
throughput.

Output in `benchmarks/results/` (git-ignored):
- `V0_<device>.csv`: one row per (frame, pass, stage):
  `variant,device,precision,frame,pass,stage,ms,n_points,n_pillars,n_fg_points`.
  The C++ `benchmark.cpp` writes the same columns.
- `V0_<device>_meta.json`: git commit, date, driver, torch/CUDA versions, power mode (Jetson).

At the end it prints the summary table (median ms per stage, e2e, FPS) and the ego-motion error
against the ground truth (a few cm; if it's much larger, something broke).

`precision` is `tf32` by default: like TensorRT, PyTorch uses TF32 for convolutions on Ampere and
newer. Use `--no-tf32` for strict fp32.

Useful options:

```bash
python benchmarks/bench_v0.py --frames 20 --passes 1     # quick run
python benchmarks/bench_v0.py --help                     # all options
```

For stable numbers: laptop plugged in, performance power profile, nothing else using the GPU.

## 3. Compare runs

```bash
python benchmarks/summarize.py benchmarks/results/*.csv
python benchmarks/summarize.py benchmarks/results/*.csv --out benchmark_results.csv
```

One column per (device, variant, precision), with the speedup against `V0` (`--baseline` to pick
another run). The script only needs numpy, so it also runs on the Jetson.

## 4. Check that outputs didn't change

Save the outputs of the original model once:

```bash
python benchmarks/bench_v0.py --deterministic --passes 1 --save-outputs benchmarks/results/V0_outputs
```

Per frame, this writes `points.bin` (input after preprocessing), `poses.bin` `[5,4,4]`,
`mos_label.bin` `[N]` and `inst_label.bin` `[N]`. The format is in `benchmarks/binio.py`
(int32 dtype code, int32 ndim, int32 dims, then the data); the C++ program writes the same files.

After a change, compare against them:

```bash
python benchmarks/bench_v0.py --deterministic --passes 1 --compare benchmarks/results/V0_outputs
```

It prints the maximum pose difference, the % of identical MOS labels and the adjusted Rand index
of the instance labels. Unchanged code gives identical labels, ARI 1 and pose differences of
~0.002 cm.

Always use `--deterministic` for these two commands. Without it, two identical fp32 runs already
differ slightly (cuDNN algorithm choice, scatter atomics), and occasionally a point switches
instance cluster. For timing (step 2), leave it off.

## Watching GPU usage

Open a monitor in a second terminal while the benchmark runs
(e.g. `python benchmarks/bench_v0.py`):

```bash
nvidia-smi dmon                 # built in: one line per second (usage %, memory, power, temperature)
watch -n 1 nvidia-smi           # built in: full status every second
nvtop                           # htop-like; install with: sudo apt install nvtop
nvitop                          # htop-like, no sudo; install with: pip install --user nvitop
```

## Known limitations

- Mixed precision in PyTorch (`torch.autocast` fp16) fails: `models/motionnet.py`
  (`full_mos[fb_mask] = mos`) mixes float32/float16. Irrelevant for TensorRT FP16.

## With the full dataset

If you do download the full data, the README's commands work unchanged in this environment, e.g.:

```bash
python main.py configs/waymo/waymo.yaml 2 1 --misc.mode=val --misc.pretrain=checkpoints/waymo.pth \
    --path.dataset_base_local=$YOUR_DATASET_FOLDER
```

(Batch size 2 instead of the README's 10, to fit in 8 GB of GPU memory.)
