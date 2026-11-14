# Running PCAccumulation (no dataset needed)

How to test, benchmark and regression-check the pretrained model on synthetic data.
To set up the environment first, see [INSTALL.md](INSTALL.md).

## 0. Activate the environment

Do this once in every new terminal:

```bash
cd ~/Desktop/PCAccumulation
source venv_pcaccumulation/bin/activate
```

## 1. Quick check that everything works (a few seconds)

```bash
python dummy-test.py
```

You should see two ✅ lines: the checkpoint loads onto the GPU, and torchsparse works.

## 2. Run the model on synthetic scenes and time it

```bash
python synthetic_benchmark.py
```

This generates synthetic 5-sweep LiDAR scenes: ground, building facades, poles, trees, parked and
moving cars, and a moving ego vehicle. It runs them through the **real** preprocessing (crop,
ground removal, 4D voxelisation, `collate_fn`) and the pretrained Waymo model in inference
(`test`) mode. At the end it prints:

- the time per stage (preprocessing, host→GPU copy, model forward, end-to-end)
- peak GPU memory
- the model's output tensors and their shapes
- the ego-motion error against the synthetic ground truth, which should be roughly **0.05–0.09 m**.
  That shows the weights and pipeline actually work, not only that they run.

Useful options:

```bash
python synthetic_benchmark.py --iters 50                 # more runs = more stable timing
python synthetic_benchmark.py --points-per-sweep 60000   # denser scenes
python synthetic_benchmark.py --batch-size 2             # several scenes at once
python synthetic_benchmark.py --dump-npz synth_data/     # also save scenes as dataset-format .npz
python synthetic_benchmark.py --help                     # all options
```

Baseline on an RTX 4060 Laptop (fp32, batch 1, ~54k points / ~9.4k 4D pillars after
preprocessing): **~160–190 ms** forward, ~660 MB peak GPU memory.

## 3. Before optimizing: save a reference

Do this once, with the original, unmodified model:

```bash
python synthetic_benchmark.py --deterministic --save-ref ref.pt
```

This stores the inputs and outputs of the first batch.

## 4. After each change to the model: compare against the reference

```bash
python synthetic_benchmark.py --deterministic --compare ref.pt
```

The comparison reuses the exact same inputs stored in `ref.pt`.

- If it only prints `worst relative float difference` at about **1e-4**, the outputs haven't
  changed. That is the noise floor of identical runs.
- Any `DIFF`, `SHAPE` or `MISSING` lines show exactly which outputs your change affected.

Always use `--deterministic` for steps 3 and 4. Without it, two identical fp32 runs already
differ slightly (cuDNN algorithm choice, scatter atomics). Occasionally a point even switches
instance cluster, which changes the shape of `sub_rec_est`/`inst_labels_adjusted`. For timing
(step 2), leave it off, since deterministic mode can be slower.

## Watching GPU usage

Open a monitor in a second terminal while the benchmark runs
(e.g. `python synthetic_benchmark.py --iters 200`):

```bash
nvidia-smi dmon                 # built in: one line per second (usage %, memory, power, temperature)
watch -n 1 nvidia-smi           # built in: full status every second
nvtop                           # htop-like; install with: sudo apt install nvtop
nvitop                          # htop-like, no sudo; install with: pip install --user nvitop
```

## Known limitations

- `--fp16` currently fails: `models/motionnet.py:224` (`full_mos[fb_mask] = mos`) mixes
  float32/float16. Mixed precision needs fixes like this in the model code.
- Synthetic scenes are only for smoke tests, latency and regression checks. Accuracy numbers
  (IoU etc.) still need real data. A few real validation scenes are enough for that, not the
  whole `data.zip` archive.

## With the real dataset

If you do download the data, the README's commands work unchanged in this environment, e.g.:

```bash
python main.py configs/waymo/waymo.yaml 2 1 --misc.mode=val --misc.pretrain=checkpoints/waymo.pth \
    --path.dataset_base_local=$YOUR_DATASET_FOLDER
```

(Batch size 2 instead of the README's 10, to fit in 8 GB of GPU memory.)
