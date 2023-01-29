import numpy as np
from progress_bar import progress_bar_iter
from nuscenes import NuScenes
from nuscenes.utils import splits
from pathlib import Path
from nuscene import NuScenesDataset
import os, sys, pickle
from bbox_utils import center_to_corner_box3d, points_in_rbbox, corners_to_lines
from utils import makedirs, save_pkl, setup_seed, natural_key, load_pkl, to_o3d_pcd, vis_o3d, mp_process, get_blue, to_o3d_vec, get_yellow
import multiprocessing as mp
from tqdm import tqdm
from glob import glob
from pyquaternion import Quaternion
from torchsparse.utils import sparse_quantize
from remove_ground import get_non_ground

color_dict = load_pkl('distinct_colors.pkl')[99][1:]
color_dict = np.clip(color_dict, 0,1)
setup_seed(0)

string_mapper = {
    "noise": 'noise',
    "human.pedestrian.adult": 'human',
    "human.pedestrian.child": 'human',
    "human.pedestrian.wheelchair": 'human',
    "human.pedestrian.stroller": 'human',
    "human.pedestrian.personal_mobility": 'human',
    "human.pedestrian.police_officer": 'human',
    "human.pedestrian.construction_worker": 'human',
    "animal": 'animal',
    "vehicle.car": 'vehicle',
    "vehicle.motorcycle": 'vehicle',
    "vehicle.bicycle": 'vehicle',
    "vehicle.bus.bendy": 'vehicle',
    "vehicle.bus.rigid": 'vehicle',
    "vehicle.truck": 'vehicle',
    "vehicle.construction": 'vehicle',
    "vehicle.emergency.ambulance": 'vehicle',
    "vehicle.emergency.police": 'vehicle',
    "vehicle.trailer": 'vehicle',
    "movable_object.barrier": 'barrier',
    "movable_object.trafficcone": 'trafficcone',
    "movable_object.pushable_pullable": 'push/pullable',
    "movable_object.debris": 'debris',
    "static_object.bicycle_rack": 'bicycle_racks',
    "flat.driveable_surface": 'driveable',
    "flat.sidewalk": 'sidewalk',
    "flat.terrain": 'terrain',
    "flat.other": 'flat.other',
    "static.manmade": 'manmade',
    "static.vegetation": 'vegetation',
    "static.other": 'static.other',
    "vehicle.ego": "ego"
}

def read_nuscene_bin(path):
    """
    Return:     [N, 4], xyz, timestamp
    """
    points = np.fromfile(str(path), dtype=np.float32,count=-1).reshape([-1, 5])
    points = points[:,[0,1,2,4]]
    return points

def create_split():
    """
    save the scene tokens
    """
    version = 'v1.0-trainval'
    sample_every_n_scenes = 2
    root_path = f'/scratch3/nuScenes/{version}'
    for split in ['train','val']:
        path_metainfo = os.path.join(root_path,f'infos_{split}.pkl')

        # 1. load all metainfo
        with open(path_metainfo,'rb') as f:
            sample_infos = pickle.load(f)['infos']

        # 2. sample a few scenes
        scene_tokens = [ele['scene_token'] for ele in sample_infos]
        set_scene_tokens = list(set(scene_tokens))
        np.random.shuffle(set_scene_tokens)
        sel_scene_tokens = set_scene_tokens[::sample_every_n_scenes]
        np.savetxt(f'configs/datasets/nuscenes/{split}.txt',sel_scene_tokens, fmt='%s')


def create_trainval_metainfo():
    version = 'v1.0-trainval'
    for split in ['train','val']:
        files = glob(f'/scratch2/shengyu/datasets/nuscene/customise/{split}/*/lidar/*.npy')
        files = [ele.replace('/scratch2/shengyu/datasets/nuscene/customise','') for ele in files]
        np.savetxt(f'configs/datasets/nuscenes/meta_{split}.txt',files, fmt='%s')


def get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes

def fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    for sample in progress_bar_iter(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)
        assert Path(lidar_path).exists(), (
            "you must download all trainval data, key-frame only dataset performs far worse than sweeps."
        )
        info = {
            "lidar_path": lidar_path,
            "cam_front_path": cam_path,
            "token": sample["token"],
            "sweeps": [],
            "lidar2ego_translation": cs_record['translation'],
            "lidar2ego_rotation": cs_record['rotation'],
            "ego2global_translation": pose_record['translation'],
            "ego2global_rotation": pose_record['rotation'],
            "timestamp": sample["timestamp"],
            "scene_token": sample['scene_token']
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
                cs_record = nusc.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path = nusc.get_sample_data_path(sd_rec['token'])
                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": sd_rec['token'],
                    "lidar2ego_translation": cs_record['translation'],
                    "lidar2ego_rotation": cs_record['rotation'],
                    "ego2global_translation": pose_record['translation'],
                    "ego2global_rotation": pose_record['rotation'],
                    "timestamp": sd_rec["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else:
                break
        info["sweeps"] = sweeps
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations])
        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos


def create_nuscene_dataset(version):
    """
    Generate train/val pickle
    """
    test = False
    max_sweeps=10
    root_path = f'/scratch3/nuScenes/{version}'
    nusc = NuScenes(version=version, dataroot=f'/scratch3/nuScenes/{version}', verbose=True)
    #nusc = NuScenes(version='v1.0-trainval', dataroot='/scratch3/nuScenes/v1.0-trainval', verbose=True)
    nusc.list_lidarseg_categories(sort_by='count')
    n_samples = len(nusc.sample)
    print(f'We have in total {n_samples} sample\n')
    n_scenes = len(nusc.scene)
    print(f'We have in total {n_scenes} scenes\n')

    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val

    available_scenes = get_available_scenes(nusc)

    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])
    print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    metadata = {
        "version": version,
    }

    print(f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}")
    data = {
        "infos": train_nusc_infos,
        "metadata": metadata,
    }
    with open(os.path.join(root_path, 'infos_train.pkl'), 'wb') as f:
        pickle.dump(data, f)
    data["infos"] = val_nusc_infos
    with open(os.path.join(root_path, 'infos_val.pkl'), 'wb') as f:
        pickle.dump(data, f)


def create_metainfo():
    create_nuscene_dataset('v1.0-mini')
    create_nuscene_dataset('v1.0-trainval')


def process_each_scene(info_dict):
    scene_token = info_dict['scene_token']
    infos = info_dict['infos']
    root_path = info_dict['root_path']
    version = info_dict['version']
    data_root = os.path.join(root_path, scene_token, 'lidar')
    label_root = os.path.join(root_path, scene_token, 'label')
    makedirs(data_root)
    makedirs(label_root)

    # 1. sort samples by tiemstamp
    all_scene_tokens = np.array([ele['scene_token'] for ele in infos])
    sel_samples = all_scene_tokens == scene_token
    sel_infos = np.array(infos)[sel_samples].tolist()
    sel_infos = list(sorted(sel_infos,key=lambda e: e['timestamp']))
    
    # 2. process eachsample
    for c_idx, eachsample in enumerate(sel_infos):
        c_idx = str(c_idx).zfill(3)
        data_path = os.path.join(data_root, c_idx)
        label_path = os.path.join(label_root, c_idx+'.pkl')

        if(not os.path.exists(label_path)):
            # 1. take keyframe LiDAR points
            lidar_path = eachsample['lidar_path']
            lidar_points = read_nuscene_bin(lidar_path)
            ts = eachsample["timestamp"] / 1e6
            ts_index = np.zeros((lidar_points.shape[0],1)).astype(np.int)
            lidar_points = np.hstack((lidar_points, ts_index))

            # 2. take history sweeps
            sweeps_info = eachsample['sweeps']
            sweep_list = [lidar_points]
            if(len(sweeps_info)):
                for idx, eachsweep in enumerate(sweeps_info):
                    lidar_path = eachsweep['lidar_path']
                    points_sweep = read_nuscene_bin(lidar_path)
                    sweep_ts = eachsweep["timestamp"] / 1e6
                    points_sweep[:, :3] = points_sweep[:, :3] @ eachsweep["sweep2lidar_rotation"].T
                    points_sweep[:, :3] += eachsweep["sweep2lidar_translation"]
                    points_sweep[:, 3] = ts - sweep_ts
                    ts_index = np.ones((points_sweep.shape[0],1)).astype(np.int) * (idx+1)
                    points_sweep = np.hstack((points_sweep, ts_index))
                    sweep_list.append(points_sweep)

            # x, y, z, ts_diff, ts_index
            sweep_points = np.concatenate(sweep_list, axis=0)

            # 3. assign points to bbox
            bbox = eachsample['gt_boxes']  # [n, 7]
            indices = points_in_rbbox(sweep_points[:,:3], bbox).astype(np.int)  
            indices = np.hstack([np.ones((indices.shape[0],1)) * 0.5,indices])
            ind_bbox = indices.argmax(1)[:,None] # [0 means not in any box] 
            
            # 4. save lidar and label infomation
            data = np.hstack((sweep_points, ind_bbox))
            sample_token = eachsample['token'] 
            lidarseg_path = f'/scratch3/nuScenes/{version}/lidarseg/{version}/{sample_token}_lidarseg.bin'
            assert os.path.exists(lidar_path)
            eachsample['lidarseg_path'] = lidarseg_path
            save_pkl(eachsample, label_path)
            np.save(data_path, data)
    
    print(f'{scene_token} is done')

def mp_process_nuscene():
    """
    create_split()
    mp_process()
    create_trainval_metainfo()
    """
    version = 'v1.0-trainval'
    root_path = f'/scratch3/nuScenes/{version}'
    all_sampled_scenes = []
    for split in ['train','val']:
        path_metainfo = os.path.join(root_path,f'infos_{split}.pkl')

        # 1. load all metainfo
        with open(path_metainfo,'rb') as f:
            sample_infos = pickle.load(f)['infos']

        # 2. load scene tokens
        sel_scene_tokens = np.loadtxt(f'configs/datasets/nuscenes/{split}.txt',dtype=str)

        for eachscene in sel_scene_tokens:
            infos = {
                'version': version,
                'root_path': f'/scratch2/shengyu/datasets/nuscene/customise/{split}',
                'scene_token': eachscene,
                'infos': sample_infos
            }
            all_sampled_scenes.append(infos)
    
    mp_process(process_each_scene, all_sampled_scenes)


def check_accumulated_scenes():
    path = '/scratch2/shengyu/datasets/nuscene/train'
    voxel_size = 0.1
    scenes_names = os.listdir(path)
    np.random.shuffle(scenes_names)
    for eachscene in scenes_names:
        files = sorted(glob(f'{path}/{eachscene}/lidar/*.npy'), key=natural_key)
        
        points = []
        for eachfile in files:
            pts = np.load(eachfile)[:,:3]
            labelpath = eachfile.replace('lidar','label').replace('npy','pkl')
            info = load_pkl(labelpath)

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = np.array(info["lidar2ego_translation"])[None]
            e2g_r = info["ego2global_rotation"]
            e2g_t = np.array(info["ego2global_translation"])[None]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            pts = np.dot(l2e_r_mat, pts.T).T
            pts += l2e_t
            pts = np.dot(e2g_r_mat, pts.T).T
            pts += e2g_t

            sel, _ = sparse_quantize(np.round(pts / voxel_size),return_index=True,return_invs=True)
            points.append(pts[sel])
        
        points = np.concatenate(points)
        sel, _ = sparse_quantize(np.round(points / voxel_size), return_index = True, return_invs = True)
        points = points[sel]
        points -= points.mean(0)
        pcd = to_o3d_pcd(points)
        pcd.paint_uniform_color(get_blue())
        vis_o3d([pcd], render=True, window_name=eachscene)

def check_single_frame():
    files = glob('/scratch2/shengyu/datasets/nuscene/customise/train/*/lidar/*.npy')
    np.random.shuffle(files)
    bbox_pts_threshold  = 0
    for eachfile in files:
        data = np.load(eachfile)
        # # 1. remove ground
        # is_not_ground = get_non_ground(data[:,:3])
        # data = data[is_not_ground]

        pts = data[:,:3]
        pcd = to_o3d_pcd(pts)
        ts_idx = data[:,4]

        ind_bbox = data[:,-1].astype(np.int)
        colors = color_dict[ind_bbox % 98]
        colors[ind_bbox==0] = get_blue()
        pcd.colors = to_o3d_vec(colors)

        labelpath = eachfile.replace('lidar','label').replace('npy','pkl')
        info = load_pkl(labelpath)

        bbox = info['gt_boxes']  # [n, 7]
        bbox_names = info['gt_names']
        bbox_names = np.array([string_mapper[ele] for ele in bbox_names.tolist()])
        bbox_velocity = info['gt_velocity']
        bbox_num_pts = info['num_lidar_pts']

        mask_num = bbox_num_pts > bbox_pts_threshold
        mask_name = np.array([True if ele in ['human','vehicle','animal'] else False for ele in bbox_names.tolist()])
        bbox_mask = np.logical_and(mask_name, mask_num)

        # assign fb labels
        mask = np.concatenate([np.array([False]), bbox_mask])
        fb_label = mask[ind_bbox].astype(np.int)
        fb_invalid_label = np.logical_and(ts_idx > 0, fb_label == 0 )
        fb_label[fb_invalid_label] = 2
        colors[fb_label==0] = get_blue()
        colors[fb_label==1] = get_yellow()
        colors[fb_label == 2] = [1,0,0]
        pcd.colors = to_o3d_vec(colors)

        # assign sd labels
        norm_velocity = np.linalg.norm(bbox_velocity, axis=1)
        velocity_threshold = 1.0
        dynamic_mask = norm_velocity > velocity_threshold
        dynamic_mask = np.concatenate([np.array([False]), dynamic_mask])
        sd_label = dynamic_mask[ind_bbox].astype(np.int)
        colors[sd_label==0] = get_blue()
        colors[sd_label==1] = get_yellow()
        pcd.colors = to_o3d_vec(colors)

        # 5. explore static/dynamic information
        # if(mask_dynamic.sum()):
        #     mask = mask_dynamic
        #     bbox_velocity, bbox_names, bbox = bbox_velocity[mask], bbox_names[mask], bbox[mask]


        bbox_velocity, bbox_names, bbox = bbox_velocity[bbox_mask], bbox_names[bbox_mask], bbox[bbox_mask]
        corner = center_to_corner_box3d(bbox[:,:3],bbox[:,3:6],bbox[:,-1])
        bbox_visuals = []
        for idx in range(corner.shape[0]):
            bbox_visuals.append(corners_to_lines(corner[idx]))
        
        bbox_visuals.append(pcd)
        n_sweeps = len(info['sweeps'])
        vis_o3d(bbox_visuals,render = True, window_name=f'{n_sweeps} sweeps')




if __name__=='__main__':
    #check_accumulated_scenes()
    # check_single_frame()
    create_split()
    mp_process_nuscene()
    create_trainval_metainfo()