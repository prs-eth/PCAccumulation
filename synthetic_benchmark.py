"""Dataset-free benchmark / regression harness for PCAccumulation.

Generates synthetic 5-sweep LiDAR scenes (ground, walls, poles, parked + moving cars, moving ego
vehicle), runs them through the REAL preprocessing (crop, ground removal, 4D voxelisation,
collate_fn) and the pretrained MotionNet in inference ('test') mode, and reports per-stage latency
and GPU memory.

Use --save-ref / --compare to check that an optimized model (fp16, TensorRT, pruned, ...) still
produces the same outputs as the original one on identical inputs.

Examples:
  python synthetic_benchmark.py                                  # benchmark, batch 1
  python synthetic_benchmark.py --points-per-sweep 60000 --iters 50
  python synthetic_benchmark.py --save-ref ref.pt                # store reference outputs
  python synthetic_benchmark.py --compare ref.pt                 # diff against reference
  python synthetic_benchmark.py --dump-npz synth_data/           # write dataset-format .npz files
"""
import argparse, os, time, copy
import numpy as np
import torch

from toolbox.config import get_config
from libs.dataset import WaymoDataset
from libs.dataloader import collate_fn
from libs.voxel_generator import Voxelization
from models.motionnet import MotionNet


# ----------------------------------------------------------------------------- synthetic scene
def _yaw_pose(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def _box_surface(rng, n, l, w, h):
    """Points sampled on the surface of an axis-aligned box centred at origin, bottom at z=0."""
    faces = rng.integers(0, 5, n)              # 4 sides + roof (no bottom)
    u, v = rng.random(n), rng.random(n)
    p = np.zeros((n, 3))
    p[:, 0] = (u - 0.5) * l
    p[:, 1] = (v - 0.5) * w
    p[:, 2] = rng.random(n) * h
    p[faces == 0, 0] = -l / 2
    p[faces == 1, 0] = l / 2
    p[faces == 2, 1] = -w / 2
    p[faces == 3, 1] = w / 2
    p[faces == 4, 2] = h
    return p


def make_scene(seed, n_frames=5, points_per_sweep=40000, dt=0.1):
    rng = np.random.default_rng(seed)
    ego_speed = rng.uniform(0, 15)             # m/s
    ego_yaw_rate = rng.uniform(-0.2, 0.2)      # rad/s
    # sensor->world pose for frame t (t=0 is the current/reference frame, t>0 are past sweeps)
    ego_poses = []
    for t in range(n_frames):
        tau = -t * dt
        yaw = ego_yaw_rate * tau
        ego_poses.append(_yaw_pose(ego_speed * tau * np.cos(yaw / 2), ego_speed * tau * np.sin(yaw / 2), yaw))
    ego_motion_gt = np.stack([np.linalg.inv(ego_poses[0]) @ P for P in ego_poses])  # frame t -> frame 0

    # static world: ground, two walls, poles, parked cars
    static_parts = []
    n_ground = int(points_per_sweep * 0.45)
    r = 32 * np.sqrt(rng.random(n_ground) ** 1.8)
    th = rng.random(n_ground) * 2 * np.pi
    static_parts.append(np.c_[r * np.cos(th), r * np.sin(th), rng.normal(0, 0.02, n_ground)])
    # building facades: segments with gaps, setbacks and varying heights (uniform walls would make
    # translation along the street unobservable, a degenerate case for registration)
    n_wall = int(points_per_sweep * 0.24)
    for side in (-1, 1):
        x = -45.0
        while x < 45:
            length, gap = rng.uniform(4, 15), rng.uniform(1, 6)
            y0, height = side * rng.uniform(9, 18), rng.uniform(3, 12)
            n = int(n_wall / 2 * length / 60)
            static_parts.append(np.c_[rng.uniform(x, x + length, n), np.full(n, y0) + rng.normal(0, 0.03, n),
                                      rng.uniform(0, height, n)])
            m = n // 4                                   # side face of the segment
            static_parts.append(np.c_[np.full(m, x), rng.uniform(min(y0, y0 + side * 5), max(y0, y0 + side * 5), m),
                                      rng.uniform(0, height, m)])
            x += length + gap
    for _ in range(rng.integers(15, 30)):                # poles and tree trunks/crowns
        n, px, py = 150, rng.uniform(-30, 30), rng.choice([-1, 1]) * rng.uniform(6, 9)
        static_parts.append(np.c_[np.full(n, px), np.full(n, py), rng.uniform(0, 4, n)] + rng.normal(0, 0.05, (n, 3)))
        if rng.random() < 0.5:
            static_parts.append(np.array([px, py, 4.5]) + rng.normal(0, 1.0, (300, 3)))
    parked = []
    for _ in range(rng.integers(4, 12)):
        pts = _box_surface(rng, 400, 4.5, 1.9, 1.6)
        T = _yaw_pose(rng.uniform(-30, 30), rng.choice([-1, 1]) * rng.uniform(4, 7), rng.normal(0, 0.1))
        parked.append((pts @ T[:3, :3].T) + T[:3, 3])
    static_world = np.concatenate(static_parts + parked)
    n_static_fg = sum(len(p) for p in parked)
    static_fb = np.r_[np.zeros(len(static_world) - n_static_fg), np.ones(n_static_fg)]
    static_inst = np.zeros(len(static_world))
    offset = len(static_world) - n_static_fg
    for i, p in enumerate(parked):
        static_inst[offset:offset + len(p)] = i + 1
        offset += len(p)

    # moving cars: constant velocity in world frame
    n_dyn = int(rng.integers(2, 6))
    dyn_cars = []
    for k in range(n_dyn):
        dyn_cars.append(dict(pts=_box_surface(rng, 600, 4.5, 1.9, 1.6),
                             x0=rng.uniform(-25, 25), y0=rng.choice([-1, 1]) * rng.uniform(0, 3.5),
                             yaw=rng.choice([0, np.pi]) + rng.normal(0, 0.05), v=rng.uniform(3, 20),
                             inst=len(parked) + k + 1))

    raw, tidx, sd, fb, inst = [], [], [], [], []
    bbox_tsfm = np.tile(np.eye(4), (n_dyn, n_frames, 1, 1))
    for t in range(n_frames):
        world_to_sensor = np.linalg.inv(ego_poses[t])
        keep = rng.random(len(static_world)) < 0.85      # each sweep sees a different subset
        parts = [(static_world[keep], np.zeros(keep.sum()), static_fb[keep], static_inst[keep])]
        for k, c in enumerate(dyn_cars):
            tau = -t * dt
            T = _yaw_pose(c['x0'] + c['v'] * tau * np.cos(c['yaw']), c['y0'] + c['v'] * tau * np.sin(c['yaw']), c['yaw'])
            T0 = _yaw_pose(c['x0'], c['y0'], c['yaw'])
            bbox_tsfm[k, t] = T0 @ np.linalg.inv(T)          # aligns car points at t to their pose at t=0
            p = (c['pts'] @ T[:3, :3].T) + T[:3, 3]
            parts.append((p, np.ones(len(p)), np.ones(len(p)), np.full(len(p), c['inst'])))
        for p, s, f, i in parts:
            ps = (p @ world_to_sensor[:3, :3].T) + world_to_sensor[:3, 3] + rng.normal(0, 0.01, p.shape)
            raw.append(ps); sd.append(s); fb.append(f); inst.append(i); tidx.append(np.full(len(p), t))

    return dict(raw_points=np.concatenate(raw).astype(np.float32),
                time_indice=np.concatenate(tidx).astype(np.float32),
                sd_labels=np.concatenate(sd).astype(np.int64),
                fb_labels=np.concatenate(fb).astype(np.int64),
                inst_labels=np.concatenate(inst).astype(np.int64),
                sem_labels=np.concatenate(fb).astype(np.int64),
                ego_motion_gt=ego_motion_gt.astype(np.float32),
                bbox_tsfm=bbox_tsfm.astype(np.float32))


# ----------------------------------------------------------------------------- helpers
def build_config(path, mode):
    cfg = get_config(path)
    cfg['config_file'] = path
    for a, b in [('voxel_size', 'voxel_size'), ('pc_range', 'range'), ('n_sweeps', 'n_sweeps')]:
        cfg['pillar_encoder'][a] = cfg['voxel_generator'][b]
    cfg['misc']['mode'] = mode
    cfg['instances'] = {'device': torch.device('cuda')}
    return cfg


def make_preprocessor(cfg):
    """WaymoDataset without the on-disk index: same prep_input(), no dataset folder needed."""
    ds = WaymoDataset.__new__(WaymoDataset)
    ds.augmentation = False
    ds.voxeliser = Voxelization(cfg['voxel_generator'])
    ds.n_frames = cfg['data']['n_frames']
    ds.crop_xy, ds.crop_z_min, ds.crop_z_max = cfg['voxel_generator']['crop_range']
    ds.remove_ground = cfg['data']['remove_ground']
    ds.ground_height = cfg['data']['ground_height'] + cfg['data']['ground_slack']
    ds.mode = cfg['misc']['mode']
    return ds


def preprocess(ds, scenes):
    samples = [ds.prep_input(s['raw_points'], s['sd_labels'], s['fb_labels'], s['inst_labels'],
                             s['time_indice'], s['ego_motion_gt'], s['bbox_tsfm']) for s in scenes]
    return collate_fn(samples)


def to_device(batch):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else
                [e.cuda() for e in v] if isinstance(v, list) and v and torch.is_tensor(v[0]) else v)
            for k, v in batch.items()}


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def flatten(obj, prefix=''):
    out = {}
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{prefix}[{i}]'))
    return out


def stats(ms):
    ms = np.asarray(ms)
    return f'mean {ms.mean():7.2f} ms | median {np.median(ms):7.2f} | p90 {np.percentile(ms, 90):7.2f} | min {ms.min():7.2f}'


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/waymo/waymo.yaml')
    ap.add_argument('--ckpt', default='checkpoints/waymo.pth')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--points-per-sweep', type=int, default=40000)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fp16', action='store_true', help='run forward under torch.autocast(float16) '
                    '(currently fails: models/motionnet.py:224 mixes float32/float16 in an index_put)')
    ap.add_argument('--deterministic', action='store_true',
                    help='cudnn.deterministic, no cudnn.benchmark, no TF32 (for --save-ref/--compare)')
    ap.add_argument('--save-ref', help='save outputs of the first batch (and its inputs) to this file')
    ap.add_argument('--compare', help='compare outputs against a file written by --save-ref')
    ap.add_argument('--dump-npz', help='also write the synthetic scenes as dataset-format .npz files here')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = build_config(args.config, 'test')
    ds = make_preprocessor(cfg)

    model = MotionNet(cfg).cuda().eval()
    state = torch.load(args.ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    print(f'Loaded {args.ckpt} (epoch {state["epoch"]}), {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params')
    print(f'GPU: {torch.cuda.get_device_name(0)} | batch {args.batch_size} | {args.points_per_sweep} pts/sweep | fp16={args.fp16}')

    n_batches = args.warmup + args.iters
    ref = torch.load(args.compare, weights_only=False) if args.compare else None
    if ref is not None:
        n_batches = max(n_batches, 1)
    t_gen, t_pre, t_h2d, t_fwd = [], [], [], []
    first_out, first_in = None, None
    n_pts, n_vox = [], []

    for b in range(n_batches):
        t0 = time.perf_counter()
        scenes = [make_scene(args.seed * 100000 + b * args.batch_size + i, cfg['data']['n_frames'], args.points_per_sweep)
                  for i in range(args.batch_size)]
        if args.dump_npz:
            os.makedirs(args.dump_npz, exist_ok=True)
            for i, s in enumerate(scenes):
                np.savez(os.path.join(args.dump_npz, f'synth_{b:04d}_{i}.npz'), **s)
        t1 = time.perf_counter()
        batch = preprocess(ds, scenes)
        if b == 0 and ref is not None:
            batch = ref['inputs']            # identical inputs for the comparison
        t2 = time.perf_counter()
        inp = to_device(copy.copy(batch))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=args.fp16):
            out = model(inp)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        if b == 0:
            first_out, first_in = to_cpu(out), batch
            torch.cuda.reset_peak_memory_stats()
        if b >= args.warmup:
            t_gen.append((t1 - t0) * 1e3); t_pre.append((t2 - t1) * 1e3)
            t_h2d.append((t3 - t2) * 1e3); t_fwd.append((t4 - t3) * 1e3)
            n_pts.append(int(batch['num_points'].sum())); n_vox.append(int(batch['num_voxels'].sum()))

    if t_fwd:
        print(f'\nInput after crop/ground removal: ~{np.mean(n_pts):.0f} points, ~{np.mean(n_vox):.0f} 4D pillars per batch')
        print(f'scene generation (synthetic only)  {stats(t_gen)}')
        print(f'preprocess (crop+voxelise+collate) {stats(t_pre)}')
        print(f'host->device copy                  {stats(t_h2d)}')
        print(f'model forward (GPU, synced)        {stats(t_fwd)}')
        print(f'end-to-end w/o scene generation    {stats(np.add(np.add(t_pre, t_h2d), t_fwd))}')
        print(f'peak GPU memory (forward)          {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')

    flat = flatten(first_out)
    print(f'\nModel outputs (first batch): {len(flat)} tensors')
    for k, v in list(flat.items())[:25]:
        print(f'  {k:45s} {str(tuple(v.shape)):22s} {v.dtype}')
    if len(flat) > 25:
        print(f'  ... {len(flat) - 25} more')
    if 'ego_motion_est' in first_out:
        est = first_out['ego_motion_est'].float()
        gt = first_in['ego_motion_gt'].float().to(est.device)
        if est.shape == gt.shape:
            print(f'ego-motion translation error vs synthetic GT: {(est[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1).mean():.3f} m '
                  '(synthetic domain gap: only a rough sanity check)')

    if args.save_ref:
        torch.save({'inputs': first_in, 'outputs': first_out, 'fp16': args.fp16}, args.save_ref)
        print(f'\nSaved reference inputs+outputs to {args.save_ref}')

    if ref is not None:
        print(f'\nComparison against {args.compare}:')
        ref_flat = flatten(ref['outputs'])
        worst = 0.0
        for k, r in ref_flat.items():
            if k not in flat:
                print(f'  MISSING  {k}'); continue
            o = flat[k]
            if o.shape != r.shape:
                print(f'  SHAPE    {k}: {tuple(r.shape)} -> {tuple(o.shape)}'); continue
            if r.dtype in (torch.bool, torch.int32, torch.int64, torch.uint8):
                mism = (o != r).float().mean().item()
                if mism > 0:
                    print(f'  DIFF     {k}: {mism * 100:.3f}% of entries differ')
            else:
                d = (o.float() - r.float()).abs().max().item() if r.numel() else 0.0
                rel = d / (r.float().abs().max().item() + 1e-12) if r.numel() else 0.0
                worst = max(worst, rel)
                if rel > 1e-3:
                    print(f'  DIFF     {k}: max abs {d:.3e} (rel {rel:.2e})')
        print(f'  worst relative float difference: {worst:.2e}')


if __name__ == '__main__':
    main()
