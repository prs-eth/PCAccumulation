"""Export the dense parts of MotionNet to ONNX for the C++ / TensorRT pipeline (Phase 0, steps 4-5).

Runs the original model once on a real selected frame (test mode). That gives real example inputs
for tracing, and the original module outputs to verify against. Then it exports each wrapper with
opset 17 and checks it in onnxruntime.

Engines (B = 1, T = 5, C = 32, H = W = 288; N points, M pillars, K foreground points):

  pfn_a     features [N, 9]                                    -> net   [N, 32]
            fc_pos + blocks[0]. Inputs = the 9 normalized per-point features
            (x, y, z, dist to pillar mean, offset to pillar center (x, y), t),
            first 8 divided by |range[0]| (36), t divided by n_sweeps (5).
  pfn_b     features [N, 64] = cat(net, pillar_max(net)[p2v])     -> net   [N, 32]
            blocks[1]
  pfn_c     features [N, 64] = cat(net, pillar_max(net)[p2v])     -> feats [N, 32]
            blocks[2] + fc_c. Then pillar_max over points gives the [M, 32] pillar features.
            NOTE: the outputs of all three are NOT post-ReLU (ResNet block = shortcut + dx) and can be
            negative. See the sign report printed by --verify.

  backbone  bev [T, C, H, W]  -> bev_feats [T, 32, H, W], fb_seg [T, 2, H, W],
                                 geo_feats [T, 64, H, W] (L2-normalized over channels)
            UNet + semseg_head + ego_feats_head.

  ego_ot    feats_s [4, 1024, 64], feats_t [4, 1024, 64], coor_s [4, 1024, 3], coor_t [4, 1024, 3],
            max_dist [4]  ->  weighted_t [4, 1024, 3], weights [4, 1024]
            Pair i (i = 0..3) registers source = frame i+1 to target = anchor frame 0 ('skip' mode).
            feats/coor are the sampled background pillars (geo_feats / pillar point mean).
            max_dist[i] = (i+1) / freq * max_speed (= 3, 6, 9, 12 m for Waymo).
            Then pose_i = weighted Kabsch(coor_s[i], weighted_t[i], weights[i]) on the CPU
            (toolbox/register_utils.kabsch_transformation_estimation: weights normalized with +1e-7).

  stpn      bev_feats [1, T, C, H, W] (backbone output), grids [T-1, H, W, 2], points [K, 3]
            -> mos [K, 2], offset [K, 2]
            grids[i] = MotionNet.get_transformed_grid(inv(pose_{i+1})), permuted to [H, W, 2].
            points = foreground points already transformed into the anchor frame.
            NOTE: the original warp_feats stacks [bev_feats[T-1], warp(frames 1..T-1)], i.e. the
            un-warped slot holds frame T-1, not the anchor frame 0 (loop variable leak in
            models/motionnet.py). The checkpoint was trained like that, so the wrapper reproduces it.

Usage:
    python export/export_onnx.py                          # export all + verify on the first frame
    python export/export_onnx.py --frame 10 --out-dir pcacc_cpp/models
    python export/export_onnx.py --only backbone stpn --no-verify
"""
import argparse, copy, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from common.reference import (build_config, load_model as load_reference_model, read_frame_list,
                              load_frame, make_preprocessor, prepare_batch)
from toolbox.utils import square_distance
from toolbox.register_utils import kabsch_transformation_estimation

ENGINES = ['pfn_a', 'pfn_b', 'pfn_c', 'backbone', 'ego_ot', 'stpn']
EPS_WEIGHTS = 1e-20       # toolbox.utils._EPS, used in weighted_t


# ============================================================================ wrappers
class PfnA(nn.Module):
    def __init__(self, pfn):
        super().__init__()
        self.fc_pos, self.block = pfn.fc_pos, pfn.blocks[0]

    def forward(self, features):
        return self.block(self.fc_pos(features))


class PfnB(nn.Module):
    def __init__(self, pfn):
        super().__init__()
        self.block = pfn.blocks[1]

    def forward(self, features):
        return self.block(features)


class PfnC(nn.Module):
    def __init__(self, pfn):
        super().__init__()
        self.block, self.fc_c = pfn.blocks[2], pfn.fc_c

    def forward(self, features):
        return self.fc_c(self.block(features))


class Backbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.unet, self.semseg_head, self.ego_feats_head = model.unet, model.semseg_head, model.ego_feats_head

    def forward(self, bev):
        bev_feats = self.unet(bev)
        fb_seg = self.semseg_head(bev_feats)
        geo = self.ego_feats_head(bev_feats)
        geo = geo / torch.norm(geo, p=2, dim=1, keepdim=True)
        return bev_feats, fb_seg, geo


def _logsumexp(x, dim):
    # max-subtracted form: identical math to torch.logsumexp, but it doesn't export to ONNX
    # ReduceLogSumExp, which TensorRT evaluates as log(sum(exp(x))) and can overflow / underflow.
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))


class EgoOT(nn.Module):
    """Same math as EgoMotionHead.pairwise_ego_motion_estimation after sampling, batched over 4 pairs."""
    def __init__(self, head):
        super().__init__()
        self.alpha, self.beta = head.alpha, head.beta
        self.softplus = nn.Softplus()
        self.n_iters = head.sinkhorn_iter

    def sinkhorn(self, log_alpha):
        la = F.pad(log_alpha, (0, 1, 0, 1))          # slack row + column of zeros
        for _ in range(self.n_iters):
            rows = la[:, :-1, :]                       # don't normalize the slack row
            la = torch.cat((rows - _logsumexp(rows, 2), la[:, -1:, :]), dim=1)
            cols = la[:, :, :-1]                       # don't normalize the slack column
            la = torch.cat((cols - _logsumexp(cols, 1), la[:, :, -1:]), dim=2)
        return la[:, :-1, :-1]

    def forward(self, feats_s, feats_t, coor_s, coor_t, max_dist):
        support = (square_distance(coor_s, coor_t, normalised=False) < (max_dist * max_dist)[:, None, None]).float()
        feat_dist = square_distance(feats_s, feats_t, normalised=True)
        affinity = -(feat_dist - self.softplus(self.alpha)) / (torch.exp(self.beta) + 0.02)
        perm = torch.exp(self.sinkhorn(affinity)) * support
        weights = torch.sum(perm, dim=2, keepdim=True)
        weighted_t = perm @ coor_t / (weights + EPS_WEIGHTS)
        return weighted_t, weights[..., 0]


class StpnExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stpn = model.motionhead
        self.sx, self.sy = abs(model.pc_range[0]), abs(model.pc_range[1])

    def forward(self, bev_feats, grids, points):
        feats = bev_feats[0]                                            # [T, C, H, W]
        T = feats.shape[0]
        warped = F.grid_sample(feats[1:], grids, mode='bilinear', padding_mode='zeros', align_corners=False)
        x = torch.cat((feats[T - 1:T], warped), dim=0)                  # frame T-1 first, see module doc
        x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()             # [1, C, T, H, W]

        s = self.stpn
        x = s.init_conv(x)
        x = torch.max(x, dim=2)[0]
        encoder_outs = []
        for module in s.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(s.up_convs):
            x = module(encoder_outs[-(i + 2)], x)                       # [1, 64, H, W]

        # ungrid: one [1, 1, K, 2] grid instead of the original [n+1, H, W, 2] blocks (same bilinear result)
        # divide by Python scalars like the original (on CUDA that is a multiply by the reciprocal;
        # dividing by a tensor differs by 1 ulp, which shows up as ~1e-5 in mos)
        uv = torch.stack([points[:, 0] / self.sx, points[:, 1] / self.sy], dim=1)
        f = F.grid_sample(x, uv.view(1, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=False)
        f = f[0, :, 0, :].t()                                            # [K, 64]

        enc = s.final_proj(torch.cat([s.positional_encoding(points / self.sx), f], dim=-1))
        return s.mos_seg(enc), s.safe_guard_offset(s.offset_head(enc))


# ============================================================================ capture
def load_model(cfg_path, ckpt):
    cfg = build_config(cfg_path, 'test')
    model = load_reference_model(cfg, ckpt)
    pfn = model.pillar_encoder
    assert len(pfn.blocks) == 3, 'pfn_a/b/c split assumes pillar_encoder depth = 3'
    assert model.ego_motion_head.seq_pose == 'skip', 'ego_ot assumes seq_pose = skip'
    assert not model.ego_motion_head.refine_with_icp, 'ICP refinement is not part of the export'
    return cfg, model


def run_and_capture(model, batch):
    """One original forward pass, recording the inputs/outputs of every exported part."""
    cap = {'ego': []}
    hooks = []

    def hook(name, keep_inputs=True):
        def fn(mod, inp, out):
            cap[name] = {'in': inp if keep_inputs else None, 'out': out}
        return fn

    hooks.append(model.pillar_encoder.register_forward_hook(hook('pfn')))
    hooks.append(model.unet.register_forward_hook(hook('unet')))
    hooks.append(model.semseg_head.register_forward_hook(hook('semseg', False)))
    hooks.append(model.ego_feats_head.register_forward_hook(hook('egofeat', False)))
    hooks.append(model.motionhead.register_forward_hook(hook('stpn')))

    head = model.ego_motion_head
    orig_pair, orig_warp = head.pairwise_ego_motion_estimation, model.warp_feats

    def pair(feats_s, feats_t, coor_s, coor_t, duration):
        rng = torch.get_rng_state()                  # to replay the exact torch.randperm draws
        pose, perm = orig_pair(feats_s, feats_t, coor_s, coor_t, duration)
        cap['ego'].append(dict(args=(feats_s, feats_t, coor_s, coor_t, duration), rng=rng, pose=pose, perm=perm))
        return pose, perm

    def warp(bev_feats, pose):
        cap['warp'] = {'bev_feats': bev_feats, 'pose': pose}
        return orig_warp(bev_feats, pose)

    head.pairwise_ego_motion_estimation, model.warp_feats = pair, warp
    try:
        with torch.no_grad():
            out = model(batch)
    finally:
        for h in hooks:
            h.remove()
        del head.pairwise_ego_motion_estimation, model.warp_feats
    cap['results'] = out
    return cap


def pfn_point_features(pfn, raw_points, p2v, coordinates, pillar_mean, time_indice):
    """The 9 normalized per-point features (what pillar_ops.cu must produce), as in PillarFeatureNet.forward."""
    mc = coordinates[p2v]
    f_center = torch.stack([raw_points[:, 0] - (mc[:, 3] * pfn.vx + pfn.x_offset),
                            raw_points[:, 1] - (mc[:, 2] * pfn.vy + pfn.y_offset)], dim=1)
    feats = torch.cat([raw_points, raw_points - pillar_mean[p2v], f_center, time_indice[:, 1:2]], dim=-1).float()
    feats[:, :-1] /= pfn.scale
    feats[:, -1] /= pfn.n_frames
    return feats


def sample_choice(n, k):
    """EgoMotionHead's sampling: randperm when n > k, else pad with the last index."""
    if n > k:
        return torch.randperm(n)[:k]
    c = torch.arange(k)
    c[n:] = n - 1
    return c


def build_example_inputs(model, cap):
    """Example (real) inputs for each wrapper plus the original outputs to compare against."""
    ex = {}
    with torch.no_grad():
        # pillar encoder
        raw_points, p2v, coords, pillar_mean, time_indice = cap['pfn']['in']
        ex['pfn'] = dict(features=pfn_point_features(model.pillar_encoder, raw_points, p2v, coords, pillar_mean,
                                                     time_indice),
                         p2v=p2v, ref=cap['pfn']['out'])
        # backbone
        bev = cap['unet']['in'][0]
        geo = cap['egofeat']['out']
        ex['backbone'] = dict(bev=bev, ref=(cap['unet']['out'], cap['semseg']['out'],
                                            geo / torch.norm(geo, p=2, dim=1, keepdim=True)))
        # ego-motion: replay the sampling of each of the 4 pairs
        head = model.ego_motion_head
        k = head.ego_n_points
        fs, ft, cs, ct, md = [], [], [], [], []
        for e in cap['ego']:
            feats_s, feats_t, coor_s, coor_t, duration = e['args']
            torch.set_rng_state(e['rng'])
            i_s = sample_choice(feats_s.size(0), k)
            i_t = sample_choice(feats_t.size(0), k)
            fs.append(feats_s[i_s]); ft.append(feats_t[i_t]); cs.append(coor_s[i_s]); ct.append(coor_t[i_t])
            md.append(duration * head.ego_max_speed)
        assert len(fs) == 4, f'expected 4 ego-motion pairs, got {len(fs)}'
        perm = torch.cat([e['perm'] for e in cap['ego']])             # [4, 1024, 1024]
        coor_t = torch.stack(ct)
        w_ref = perm.sum(2)
        ex['ego_ot'] = dict(feats_s=torch.stack(fs), feats_t=torch.stack(ft), coor_s=torch.stack(cs), coor_t=coor_t,
                            max_dist=torch.tensor(md, dtype=torch.float32, device=coor_t.device),
                            ref=(perm @ coor_t / (w_ref[..., None] + EPS_WEIGHTS), w_ref),
                            ref_poses=torch.stack([e['pose'] for e in cap['ego']]))
        # stpn
        if 'stpn' not in cap:
            ex['stpn'] = None
        else:
            bev_feats, pose = cap['warp']['bev_feats'], cap['warp']['pose']   # [1, T, C, H, W], [1, T, 4, 4]
            _, T, _, H, W = bev_feats.shape
            grids = torch.stack([model.get_transformed_grid(torch.linalg.inv(pose[0, t]), H, W, model.resolution[0],
                                                            model.resolution[1], model.pc_range[0], model.pc_range[1])
                                 for t in range(1, T)]).permute(0, 2, 3, 1).contiguous()
            points = cap['stpn']['in'][1]
            mos, offset, _ = cap['stpn']['out']
            ex['stpn'] = dict(bev_feats=bev_feats, grids=grids, points=points, ref=(mos, offset))
    return ex


# ============================================================================ export
def export_specs(model, ex):
    """name -> (wrapper, example args, input names, output names, dynamic axes)."""
    pfn = model.pillar_encoder
    with torch.no_grad():
        feats = ex['pfn']['features']
        p2v = ex['pfn']['p2v']
        a = PfnA(pfn)(feats)
        b_in = torch.cat([a, scatter(a, p2v, dim=0, reduce='max')[p2v]], dim=1)
    dyn_n = {'features': {0: 'n_points'}}
    specs = {
        'pfn_a': (PfnA(pfn), (feats,), ['features'], ['net'], {**dyn_n, 'net': {0: 'n_points'}}),
        'pfn_b': (PfnB(pfn), (b_in,), ['features'], ['net'], {**dyn_n, 'net': {0: 'n_points'}}),
        'pfn_c': (PfnC(pfn), (b_in,), ['features'], ['feats'], {**dyn_n, 'feats': {0: 'n_points'}}),
        'backbone': (Backbone(model), (ex['backbone']['bev'],), ['bev'], ['bev_feats', 'fb_seg', 'geo_feats'], None),
        'ego_ot': (EgoOT(model.ego_motion_head),
                   tuple(ex['ego_ot'][k] for k in ['feats_s', 'feats_t', 'coor_s', 'coor_t', 'max_dist']),
                   ['feats_s', 'feats_t', 'coor_s', 'coor_t', 'max_dist'], ['weighted_t', 'weights'], None),
    }
    if ex['stpn'] is not None:
        s = ex['stpn']
        specs['stpn'] = (StpnExport(model), (s['bev_feats'], s['grids'], s['points']),
                         ['bev_feats', 'grids', 'points'], ['mos', 'offset'],
                         {'points': {0: 'n_fg_points'}, 'mos': {0: 'n_fg_points'}, 'offset': {0: 'n_fg_points'}})
    return specs


def export_one(path, wrapper, args, in_names, out_names, dyn, opset):
    import onnx
    wrapper = copy.deepcopy(wrapper).cpu().eval()
    args = tuple(a.cpu() for a in args)
    with torch.no_grad():
        torch.onnx.export(wrapper, args, path, opset_version=opset, input_names=in_names, output_names=out_names,
                          dynamic_axes=dyn, do_constant_folding=True)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    return os.path.getsize(path), ops


# ============================================================================ verification
def err(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-12)


def ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])


def ort_run(sess, *inputs):
    feed = {i.name: x.detach().cpu().numpy() for i, x in zip(sess.get_inputs(), inputs)}
    return [torch.from_numpy(o) for o in sess.run(None, feed)]


class Report:
    def __init__(self, tol):
        self.tol, self.fail = tol, []

    def tensor(self, engine, what, out, ref, tol=None):
        tol = self.tol if tol is None else tol
        d, r = err(out, ref)
        ok = r <= tol
        if not ok:
            self.fail.append(f'{engine}: {what}')
        print(f'  {"ok  " if ok else "FAIL"} {what:<44} max abs {d:9.2e}  rel {r:9.2e}')

    def info(self, text):
        print(f'       {text}')


def pfn_chain(run_a, run_b, run_c, feats, p2v):
    """engines alternating with the pillar scatter-max kernels, as planned for pillar_ops.cu."""
    pool = lambda x: torch.cat([x, scatter(x, p2v, dim=0, reduce='max')[p2v]], dim=1)
    a = run_a(feats)
    b = run_b(pool(a))
    c = run_c(pool(b))
    return scatter(c, p2v, dim=0, reduce='max'), (a, b, c)


def verify(model, ex, specs, paths, rep):
    print('\n== verification (wrapper vs original on GPU; onnxruntime CPU vs wrapper on CPU, fp32) ==')
    pfn = model.pillar_encoder
    if {'pfn_a', 'pfn_b', 'pfn_c'} <= paths.keys():
        print('pfn_a/b/c (chained with torch_scatter max in between)')
        p = ex['pfn']
        with torch.no_grad():
            out, _ = pfn_chain(PfnA(pfn), PfnB(pfn), PfnC(pfn), p['features'], p['p2v'])
        rep.tensor('pfn', 'wrapper chain vs PillarFeatureNet [M, 32]', out, p['ref'])
        sa, sb, sc = (ort_session(paths[n]) for n in ['pfn_a', 'pfn_b', 'pfn_c'])
        feats, p2v = p['features'].cpu(), p['p2v'].cpu()
        o_out, (oa, ob, oc) = pfn_chain(lambda x: ort_run(sa, x)[0], lambda x: ort_run(sb, x)[0],
                                        lambda x: ort_run(sc, x)[0], feats, p2v)
        with torch.no_grad():
            cpu = [copy.deepcopy(m).cpu() for m in (PfnA(pfn), PfnB(pfn), PfnC(pfn))]
            w_out, _ = pfn_chain(*cpu, feats, p2v)
        rep.tensor('pfn', 'onnxruntime chain vs wrapper chain', o_out, w_out)
        rep.tensor('pfn', 'onnxruntime chain vs PillarFeatureNet', o_out, p['ref'], tol=1e-4)
        for n, t in zip(['pfn_a', 'pfn_b', 'pfn_c'], [oa, ob, oc]):
            rep.info(f'{n} output range [{t.min().item():.3f}, {t.max().item():.3f}], '
                     f'{(t < 0).float().mean().item() * 100:.1f} % negative')

    for name in ['backbone', 'ego_ot', 'stpn']:
        if name not in paths:
            continue
        print(name)
        wrapper, args, *_ = specs[name]
        ref = ex[name]['ref']
        with torch.no_grad():
            gpu_out = wrapper(*args)
            cpu_wrapper = copy.deepcopy(wrapper).cpu()
            cpu_args = tuple(a.cpu() for a in args)
            t0 = time.time()
            cpu_out = cpu_wrapper(*cpu_args)
        ort_out = ort_run(ort_session(paths[name]), *cpu_args)
        names = specs[name][3]
        for n, g, c, o, r in zip(names, gpu_out, cpu_out, ort_out, ref):
            rep.tensor(name, f'{n}: wrapper vs original', g, r)
            rep.tensor(name, f'{n}: onnxruntime vs wrapper', o, c)
            rep.tensor(name, f'{n}: onnxruntime vs original', o, r, tol=1e-4)
        if name == 'backbone':
            agree = (ort_out[1].argmax(1) == ref[1].cpu().argmax(1)).float().mean().item()
            rep.info(f'fb_seg argmax agreement onnxruntime vs original: {agree * 100:.4f} %')
        if name == 'ego_ot':
            e = ex['ego_ot']
            R, t, _, _ = kabsch_transformation_estimation(e['coor_s'].cpu(), ort_out[0], weights=ort_out[1])
            ref_p = e['ref_poses'].cpu()
            dt = (t[..., 0] - ref_p[:, :3, 3]).norm(dim=1).max().item()
            dr = (R - ref_p[:, :3, :3]).abs().max().item()
            ok = dt < 0.01
            if not ok:
                rep.fail.append('ego_ot: pose')
            print(f'  {"ok  " if ok else "FAIL"} {"poses (Kabsch on onnxruntime outputs)":<44} '
                  f'max trans diff {dt * 100:.4f} cm, max rot diff {dr:.2e}')
        if name == 'stpn':
            agree = (ort_out[0].argmax(1) == ref[0].cpu().argmax(1)).float().mean().item()
            rep.info(f'K = {args[2].shape[0]} points, mos argmax agreement onnxruntime vs original: {agree * 100:.4f} %')


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/waymo/waymo.yaml')
    ap.add_argument('--ckpt', default='checkpoints/waymo.pth')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--frame-list', default='data/waymo/selected_frames.txt')
    ap.add_argument('--frame', type=int, default=0, help='index in the frame list used for tracing / verification')
    ap.add_argument('--out-dir', default='pcacc_cpp/models')
    ap.add_argument('--only', nargs='+', choices=ENGINES, help='export only these engines')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-5, help='max relative error (max abs diff / max |ref|)')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()
    os.chdir(REPO)

    # strict fp32 everywhere, so the comparisons measure the export and not TF32 noise
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg, model = load_model(args.config, args.ckpt)
    rel = read_frame_list(args.frame_list)[args.frame]
    batch = prepare_batch(make_preprocessor(cfg), load_frame(args.data_root, rel))
    print(f'frame {rel}: {batch["input_points"].shape[0]} points, {int(batch["num_voxels"].sum())} pillars')

    torch.manual_seed(args.seed)
    cap = run_and_capture(model, batch)
    ex = build_example_inputs(model, cap)
    if ex['stpn'] is None:
        sys.exit(f'STPN did not run on {rel} (too few foreground points); pick another --frame')
    print(f'foreground points sent to STPN: {ex["stpn"]["points"].shape[0]}')

    specs = export_specs(model, ex)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {}
    print(f'\n== export (opset {args.opset}) -> {args.out_dir} ==')
    for name in (args.only or ENGINES):
        path = os.path.join(args.out_dir, f'{name}.onnx')
        size, ops = export_one(path, *specs[name], args.opset)
        paths[name] = path
        print(f'  {name:<9} {size / 1e6:7.2f} MB  ops: {" ".join(ops)}')

    if args.no_verify:
        return
    rep = Report(args.tol)
    verify(model, ex, specs, paths, rep)
    if rep.fail:
        print(f'\nFAILED: {", ".join(rep.fail)}')
        sys.exit(1)
    print('\nall checks passed')


if __name__ == '__main__':
    main()
