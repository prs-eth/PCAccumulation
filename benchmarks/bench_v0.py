"""V0 speed benchmark: the original PyTorch model on the selected real frames.

Writes benchmarks/results/<variant>_<precision>_<device>.csv with one row per (frame, pass, stage),
in the same schema the C++ benchmark will write, plus a matching _meta.json (precision in the name,
so tf32 and fp32/deterministic runs don't overwrite each other). Compare runs with
benchmarks/summarize.py.

Stages (same boundaries as the C++ pipeline):
    preprocess  crop + ground removal                  (CPU in V0)
    voxelize    4D pillars (numba) + collate           (CPU in V0)
    h2d         host -> device copy
    pillar_enc  pillar encoder + BEV scatter
    backbone    UNet + fg/bg head + ego feature head
    ego_motion  ego-motion head (sampling, Sinkhorn, Kabsch)
    motion_seg  warp + STPN on foreground points
    cluster     DBSCAN on moving points
    tubenet     AlignNet (out of the C++ scope, not part of e2e)
    e2e         preprocess .. cluster, sum of the synced stages
    e2e_nosync  preprocess .. cluster, separate run with no per-stage syncs (real throughput)

Examples:
    python benchmarks/bench_v0.py                                    # full benchmark
    python benchmarks/bench_v0.py --frames 20 --passes 1             # quick run
    python benchmarks/bench_v0.py --frames 20 --passes 1 --view      # quick run with the live window
    python benchmarks/bench_v0.py --deterministic --passes 1 --save-outputs benchmarks/results/V0_outputs
    python benchmarks/bench_v0.py --deterministic --passes 1 --compare benchmarks/results/V0_outputs
"""
import argparse, csv, datetime, json, os, platform, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common.reference import (build_config, load_model, read_frame_list, load_frame, frame_id,
                              make_preprocessor, preprocess_mask, crop_and_remove_ground,
                              voxelize_and_collate, to_device)
from binio import write_bin, read_bin

E2E_STAGES = ['preprocess', 'voxelize', 'h2d', 'pillar_enc', 'backbone', 'ego_motion', 'motion_seg', 'cluster']
CSV_FIELDS = ['variant', 'device', 'precision', 'frame', 'pass', 'stage', 'ms', 'n_points', 'n_pillars', 'n_fg_points']


# ----------------------------------------------------------------------------- setup
# config, model, frames and preprocessing: common/reference.py (shared with export/export_onnx.py)
def device_name():
    p = torch.cuda.get_device_properties(0)
    return f'{p.name} (sm_{p.major}{p.minor})'


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '-', s).strip('-')


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip()
    except Exception:
        return ''


# ----------------------------------------------------------------------------- timing
class StageTimer:
    """mark(stage) records the time since the previous mark. With sync=True every mark waits for
    the GPU first (per-stage latency); with sync=False only the stages in sync_at do."""
    def __init__(self, sync=True, sync_at=()):
        self.sync, self.sync_at = sync, set(sync_at)

    def start(self):
        torch.cuda.synchronize()
        self.times, self.t_start = {}, time.perf_counter()
        self.t_last = self.t_start

    def mark(self, stage):
        if self.sync or stage in self.sync_at:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.times[stage] = (now - self.t_last) * 1e3
        self.t_last = now

    def since_start(self, stage):
        """total ms from start() to the given mark (cumulative)."""
        total = 0.0
        for s, ms in self.times.items():
            total += ms
            if s == stage:
                return total
        raise KeyError(stage)


def run_frame(ds, model, f, timer, seed):
    torch.manual_seed(seed)                  # ego-motion sampling uses torch.randperm
    timer.start()
    c = crop_and_remove_ground(ds, f)
    timer.mark('preprocess')
    batch = voxelize_and_collate(ds, c)
    timer.mark('voxelize')
    inp = to_device(batch)
    timer.mark('h2d')
    model.timer = timer
    with torch.no_grad():
        out = model(inp)
    model.timer = None
    torch.cuda.synchronize()
    return batch, out


# ----------------------------------------------------------------------------- outputs
def save_outputs(d, batch, out):
    os.makedirs(d, exist_ok=True)
    write_bin(os.path.join(d, 'poses.bin'), out['ego_motion_est'][0].float().cpu().numpy())
    write_bin(os.path.join(d, 'mos_label.bin'), out['mos_est'].argmax(1).cpu().numpy())
    write_bin(os.path.join(d, 'inst_label.bin'), out['inst_labels_est'].cpu().numpy())
    write_bin(os.path.join(d, 'points.bin'), batch['input_points'].numpy())
    write_bin(os.path.join(d, 'time_indice.bin'), batch['time_indice'][:, -1].numpy().astype(np.int32))   # sweep 0..T-1


def compare_outputs(d, batch, out):
    from sklearn.metrics import adjusted_rand_score
    ref_pts = read_bin(os.path.join(d, 'points.bin'))
    pts = batch['input_points'].numpy()
    if ref_pts.shape != pts.shape or not np.array_equal(ref_pts, pts):
        return {'input_mismatch': 1}
    poses = out['ego_motion_est'][0].float().cpu().numpy()
    ref_poses = read_bin(os.path.join(d, 'poses.bin'))
    mos = out['mos_est'].argmax(1).cpu().numpy()
    inst = out['inst_labels_est'].cpu().numpy()
    return {
        'input_mismatch': 0,
        'pose_trans_diff_cm': float(np.abs(poses[:, :3, 3] - ref_poses[:, :3, 3]).max() * 100),
        'pose_rot_diff': float(np.abs(poses[:, :3, :3] - ref_poses[:, :3, :3]).max()),
        'mos_agree': float((mos == read_bin(os.path.join(d, 'mos_label.bin'))).mean()),
        'inst_ari': float(adjusted_rand_score(read_bin(os.path.join(d, 'inst_label.bin')), inst)),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/waymo/waymo.yaml')
    ap.add_argument('--ckpt', default='checkpoints/waymo.pth')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--frame-list', default='data/waymo/selected_frames.txt')
    ap.add_argument('--frames', type=int, default=0, help='use only the first N frames (0 = all)')
    ap.add_argument('--variant', default='V0')
    ap.add_argument('--passes', type=int, default=3, help='timed passes over all frames')
    ap.add_argument('--warmup', type=int, default=10, help='untimed warm-up frames')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-tf32', action='store_true', help='disable TF32 in convolutions (strict fp32)')
    ap.add_argument('--deterministic', action='store_true',
                    help='cudnn.deterministic, no cudnn.benchmark, no TF32 (for --save-outputs/--compare)')
    ap.add_argument('--save-outputs', help='write poses / labels per frame (pass 0) into this folder')
    ap.add_argument('--compare', help='compare outputs (pass 0) against a folder from --save-outputs')
    ap.add_argument('--out-dir', default='benchmarks/results')
    ap.add_argument('--view', action='store_true',
                    help='live window: input sweeps, output segmentation and statistics per frame '
                         '(benchmarks/live_view.py); frames are skipped while it draws')
    ap.add_argument('--view-pause', type=float, default=0.0,
                    help='with --view: untimed pause in seconds after each frame, to watch every frame')
    args = ap.parse_args()
    os.chdir(REPO)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True    # fixed BEV shape: let cuDNN pick the fastest kernels
    if args.deterministic or args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    precision = 'tf32' if torch.backends.cudnn.allow_tf32 else 'fp32'

    cfg = build_config(args.config, 'test')
    ds = make_preprocessor(cfg)
    model = load_model(cfg, args.ckpt)

    frames = read_frame_list(args.frame_list)
    if args.frames:
        frames = frames[:args.frames]
    dev = device_name()
    print(f'{args.variant} | {dev} | {precision} | {len(frames)} frames x {args.passes} passes | '
          f'deterministic={args.deterministic}')

    def load(rel):
        return load_frame(args.data_root, rel)

    # warm-up: numba JIT, cuDNN autotuning, allocator
    synced, nosync = StageTimer(sync=True), StageTimer(sync=False, sync_at=['cluster'])
    for i, rel in enumerate(frames[:args.warmup]):
        f = load(rel)
        run_frame(ds, model, f, synced, args.seed)
        run_frame(ds, model, f, nosync, args.seed)
    print(f'warm-up done ({min(args.warmup, len(frames))} frames)')

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f'{args.variant}_{precision}_{slug(dev)}')
    rows, cmp, trans_err = [], [], []
    live = None
    if args.view:
        from live_view import LiveViewClient, frame_message
        live = LiveViewClient((args.variant, precision, dev), [frame_id(r) for r in frames], args.view_pause)
    t_bench = time.time()
    for p in range(args.passes):
        for i, rel in enumerate(frames):
            if live:
                live.wait_if_paused()        # untimed; Pause / Step from the window
            f = load(rel)                    # disk I/O is not timed
            fid = frame_id(rel)
            batch, out = run_frame(ds, model, f, synced, args.seed)
            st = dict(synced.times)
            _ = run_frame(ds, model, f, nosync, args.seed)
            st['e2e'] = sum(st[s] for s in E2E_STAGES)
            st['e2e_nosync'] = nosync.since_start('cluster')
            sizes = dict(n_points=int(batch['num_points'].sum()), n_pillars=int(batch['num_voxels'].sum()),
                         n_fg_points=int(out['fb_est_per_points'].sum()))
            for stage, ms in st.items():
                rows.append(dict(variant=args.variant, device=dev, precision=precision, frame=fid, pass_=p,
                                 stage=stage, ms=f'{ms:.3f}', **sizes))
            if p == 0:
                if 'ego_trans_error' in out:
                    trans_err.append(float(torch.as_tensor(out['ego_trans_error']).float().mean()))
                if args.save_outputs:
                    save_outputs(os.path.join(args.save_outputs, fid.replace('/', '_')), batch, out)
                if args.compare:
                    cmp.append(compare_outputs(os.path.join(args.compare, fid.replace('/', '_')), batch, out))
            if live and live.alive:          # untimed: after both runs of the frame
                live.send(frame_message(
                    fid=fid, pass_idx=p, passes=args.passes, frame_idx=i, n_frames=len(frames), stages=st,
                    sizes=(sizes['n_points'], sizes['n_pillars'], sizes['n_fg_points']),
                    raw_points=f['raw_points'], raw_time=f['time_indice'],
                    keep=preprocess_mask(ds, f['raw_points']),
                    crop=(ds.crop_xy, ds.crop_z_min, ds.crop_z_max), points=batch['input_points'].numpy(),
                    time_indice=batch['time_indice'][:, -1].numpy(), mos_label=out['mos_est'].argmax(1).cpu().numpy(),
                    inst_label=out['inst_labels_est'].cpu().numpy(),
                    poses=out['ego_motion_est'][0].float().cpu().numpy()))
            if (i + 1) % 25 == 0 or i + 1 == len(frames):
                print(f'  pass {p + 1}/{args.passes}  frame {i + 1}/{len(frames)}  e2e {st["e2e"]:.1f} ms')

    with open(base + '.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            r['pass'] = r.pop('pass_')
            w.writerow(r)

    meta = {
        'variant': args.variant, 'device': dev, 'precision': precision,
        'date': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': run_cmd(['git', 'rev-parse', '--short', 'HEAD']),
        'git_dirty': bool(run_cmd(['git', 'status', '--porcelain', '--untracked-files=no'])),
        'host': platform.node(), 'cpu': platform.processor() or platform.machine(),
        'driver': run_cmd(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
        'power_mode': run_cmd(['nvpmodel', '-q']) or None,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'frames': len(frames), 'passes': args.passes, 'warmup': args.warmup,
        'deterministic': args.deterministic, 'seed': args.seed, 'live_view': args.view,
        'benchmark_wall_s': round(time.time() - t_bench, 1),
        'ego_trans_error_vs_gt_m': float(np.mean(trans_err)) if trans_err else None,
    }
    with open(base + '_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'\nwrote {base}.csv and {base}_meta.json')
    if trans_err:
        print(f'ego-motion translation error vs GT (sanity): {np.mean(trans_err):.3f} m')

    if cmp:
        bad = sum(c['input_mismatch'] for c in cmp)
        ok = [c for c in cmp if not c['input_mismatch']]
        print(f'\ncompare vs {args.compare}: {len(ok)} frames ({bad} with different input points)')
        if ok:
            print(f'  max pose translation diff  {max(c["pose_trans_diff_cm"] for c in ok):.4f} cm')
            print(f'  max pose rotation diff     {max(c["pose_rot_diff"] for c in ok):.2e}')
            print(f'  min / mean MOS agreement   {min(c["mos_agree"] for c in ok) * 100:.3f} % / '
                  f'{np.mean([c["mos_agree"] for c in ok]) * 100:.3f} %')
            print(f'  min / mean instance ARI    {min(c["inst_ari"] for c in ok):.4f} / '
                  f'{np.mean([c["inst_ari"] for c in ok]):.4f}')

    from summarize import summarize
    table = summarize([base + '.csv'])
    if live:
        r = table[0] if table else {}
        live.finish(f"{len(frames)} frames x {args.passes} passes\ne2e median {r.get('e2e_ms')} ms, "
                    f"p90 {r.get('e2e_p90_ms')} ms, {r.get('fps')} FPS\nwrote {os.path.basename(base)}.csv")


if __name__ == '__main__':
    main()
