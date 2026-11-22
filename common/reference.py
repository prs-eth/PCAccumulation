"""The original Python model (V0) as the tools outside the upstream code use it.

Shared by benchmarks/bench_v0.py (speed benchmark, reference outputs) and export/export_onnx.py
(ONNX export + verification), so both run exactly the same config, checkpoint, frames and
preprocessing:

  config      build_config()                      configs/waymo/waymo.yaml + derived fields
  model       load_model()                        MotionNet + checkpoints/waymo.pth on the GPU
  frames      read_frame_list(), load_frame(), frame_id()
  preprocess  make_preprocessor(), preprocess_mask(), crop_and_remove_ground(),
              voxelize_and_collate(), to_device(), prepare_batch()

The preprocessing reproduces BaseDataset.prep_input (libs/dataset.py) without augmentation, split
into crop/ground removal and voxelisation so the two parts can be timed separately.

Callers must have the repository root on sys.path (for libs/, toolbox/, models/).
"""
import os

import numpy as np
import torch

from toolbox.config import get_config
from libs.dataset import WaymoDataset
from libs.dataloader import collate_fn
from libs.voxel_generator import Voxelization
from models.motionnet import MotionNet


# ----------------------------------------------------------------------------- config + model
def build_config(path, mode='test'):
    cfg = get_config(path)
    cfg['config_file'] = path
    for a, b in [('voxel_size', 'voxel_size'), ('pc_range', 'range'), ('n_sweeps', 'n_sweeps')]:
        cfg['pillar_encoder'][a] = cfg['voxel_generator'][b]
    cfg['misc']['mode'] = mode
    cfg['instances'] = {'device': torch.device('cuda')}
    return cfg


def load_model(cfg, ckpt):
    """MotionNet with the trained weights (state_dict of the training checkpoint), eval mode, GPU."""
    model = MotionNet(cfg).cuda().eval()
    state = torch.load(ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    return model


# ----------------------------------------------------------------------------- frames
def read_frame_list(path):
    """Selected frames, one path per line relative to the data root ('/waymo/test/000/0005.npz')."""
    return [l.strip() for l in open(path) if l.strip()]


def load_frame(data_root, rel):
    d = np.load(data_root + rel, allow_pickle=True)
    return {k: d[k] for k in d.files}


def frame_id(rel):
    """'/waymo/test/000/0005.npz' -> '000/0005' (as in the benchmark CSVs)."""
    parts = rel.strip('/').split('/')
    return f'{parts[-2]}/{os.path.splitext(parts[-1])[0]}'


# ----------------------------------------------------------------------------- preprocessing
def make_preprocessor(cfg):
    """WaymoDataset without the on-disk index: same prep_input() settings, no dataset folder needed."""
    ds = WaymoDataset.__new__(WaymoDataset)
    ds.augmentation = False
    ds.voxeliser = Voxelization(cfg['voxel_generator'])
    ds.n_frames = cfg['data']['n_frames']
    ds.crop_xy, ds.crop_z_min, ds.crop_z_max = cfg['voxel_generator']['crop_range']
    ds.remove_ground = cfg['data']['remove_ground']
    ds.ground_height = cfg['data']['ground_height'] + cfg['data']['ground_slack']
    ds.mode = cfg['misc']['mode']
    return ds


def preprocess_mask(ds, raw_points):
    """Points kept by crop + ground removal."""
    sel_xy = np.logical_and(np.abs(raw_points[:, 0]) < ds.crop_xy, np.abs(raw_points[:, 1]) < ds.crop_xy)
    sel_z = np.logical_and(raw_points[:, 2] < ds.crop_z_max, raw_points[:, 2] > ds.crop_z_min)
    sel = np.logical_and(sel_xy, sel_z)
    if ds.remove_ground:
        sel = np.logical_and(sel, raw_points[:, 2] > ds.ground_height)
    return sel


def crop_and_remove_ground(ds, f):
    raw_points, time_indice = f['raw_points'], f['time_indice']
    sd_labels, fb_labels, inst_labels = f['sd_labels'], f['fb_labels'], f['inst_labels']
    sel = preprocess_mask(ds, raw_points)
    return dict(raw_points=raw_points[sel], time_indice=time_indice[sel], sd_labels=sd_labels[sel],
                fb_labels=fb_labels[sel], inst_labels=inst_labels[sel],
                ego_motion_gt=f['ego_motion_gt'], inst_motion_gt=f['bbox_tsfm'])


def voxelize_and_collate(ds, c):
    raw_points, time_indice = c['raw_points'], c['time_indice']
    points = np.concatenate((raw_points, time_indice[:, None]), axis=1).astype(np.float32)
    voxels = ds.voxeliser(points)
    data = {
        'input_points': raw_points,
        'num_points': np.array([raw_points.shape[0]], dtype=np.int64),
        'time_indice': time_indice[:, None],
        'sd_labels': c['sd_labels'][:, None],
        'inst_labels': c['inst_labels'][:, None],
        'ego_motion_gt': c['ego_motion_gt'],
        'inst_motion_gt': c['inst_motion_gt'],
        'fb_labels': c['fb_labels'][:, None],
    }
    data.update(voxels)
    return collate_fn([data])


def to_device(batch):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else
                [e.cuda() for e in v] if isinstance(v, list) and v and torch.is_tensor(v[0]) else v)
            for k, v in batch.items()}


def prepare_batch(ds, f):
    """Raw frame -> model input on the GPU (all preprocessing steps, untimed)."""
    return to_device(voxelize_and_collate(ds, crop_and_remove_ground(ds, f)))
