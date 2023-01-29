import numpy as np
from libs.utils import to_o3d_pcd, vis_o3d, multi_vis, get_blue, load_pkl
from tqdm import tqdm
import torch, random

"""
369/9382 
93/2338
71/2346 are invalid samples
"""
string_mapper = {
    'car': 'vehicle',
    'truck': 'vehicle',
    'trailer': 'vehicle',
    'bus': 'vehicle',
    'construction_vehicle': 'vehicle',
    'bicycle': 'vehicle',
    'motorcycle':'vehicle',
    'pedestrian':'human',
    'traffic_cone':'static',
    'barrier':'static',
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
    Return:     [N, 5], xyz, ref, timestamp
    """
    points = np.fromfile(str(path), dtype=np.float32,count=-1).reshape([-1, 5])
    return points


def to_tensor(x):
    if isinstance(x, torch.Tensor):
      return x
    elif isinstance(x, np.ndarray):
      return torch.from_numpy(x)
    else:
      raise ValueError("Can not convert to torch tensor {}".format(x))

def reconstruct_sequence(points, time_indice, inst_labels, tsfm, n_frames):
    """
    Reconstruct a sequence of point clouds
    Input:
        points:         [N,3]
        time_indice:    [N]
        inst_labels:    [N]
        tsfm:           [M, n_frames, 4, 4]
        n_frames:       integer
    """
    inst_labels = inst_labels.long()
    assert n_frames == tsfm.size(1)

    point_tsfm = tsfm[inst_labels] #[N, n_frame, 4, 4]
    gather_indice = time_indice[:,None,None,None].repeat(1,1,4,4).long()
    point_tsfm = torch.gather(point_tsfm, dim=1, index = gather_indice).squeeze(1)        
    rot, trans = point_tsfm[:,:3,:3], point_tsfm[:,:3,3][:,:,None]
    rec_points = (torch.matmul(rot, points[:,:,None]) + trans).squeeze(-1)
    return rec_points

def ego_motion_compensation(points, time_indice, tsfm):
    """
    Input (torch.Tensor):
        points:         [N, 3]
        time_indice:    [N]
        tsfm:           [n_frames, 4, 4]
    """
    point_tsfm = tsfm[time_indice.long()]
    rot, trans = point_tsfm[:,:3,:3], point_tsfm[:,:3,3][:,:,None]
    rec_points = (torch.matmul(rot, points[:,:,None]) + trans).squeeze(-1)
    return rec_points


for split in ['train','val','test']:
    files = np.loadtxt(f'configs/datasets/nuscene/our_split/{split}_info.txt', dtype = str)
    random.shuffle(files)
    info_path = f'configs/datasets/nuscene/metadata/{split}.pkl'
    nusc_infos = load_pkl(info_path)
    tokens = [ele['token'] for ele in nusc_infos]
    token_to_seg_path_mapping = load_pkl('../spv_SpatioTempo/dataset_configs/nuscenes/nuscene_token_to_seg_path.pkl')

    count = 0
    for eachfile in tqdm(files):
        path = '/scratch3/cvpr2022/mini/'+ eachfile
        full_dict = np.load(path)
        input_points = full_dict['raw_points']
        bbox_tsfm = full_dict['bbox_tsfm']  # [K, n_frames, 4, 4]
        time_indice = full_dict['time_indice']
        ego_motion_gt = full_dict['ego_motion_gt']  # [n_frames, 4,4]
        inst_labels = full_dict['inst_labels']
        sd_labels = full_dict['sd_labels']
        fb_labels = full_dict['fb_labels']

        sel = time_indice == 1
        fb_label = fb_labels[sel]
        n_bkgd = (fb_label == 0).sum()
        n_frames = 11

        if n_bkgd < 1000:
            #print((inst_labels[sel] ==0).sum())
            scene_token = eachfile.split('/')[3]
            sample_token = eachfile.split('/')[4].split('.')[0]

            # find the token
            idx = tokens.index(sample_token)

            info = nusc_infos[idx]

            ###########################################
            # load key frame and associated lidar labels
            lidar_path = info['lidar_path'].replace('/scratch3/nuScenes','/scratch2/shengyu/datasets/nuscene')
            lidarseg_path = token_to_seg_path_mapping[info['token']]

            lidar_points = read_nuscene_bin(lidar_path)
            ts = info['timestamp'] / 1e6
            ts_index = np.zeros((lidar_points.shape[0])).astype(np.int)
            bbox, bbox_velocity = info['gt_boxes'], info['gt_velocity']
            bbox_num_pts = info['num_lidar_pts']
            bbox_names = info['gt_names']
            bbox_names = np.array([string_mapper[ele] for ele in bbox_names.tolist()])

            ###########################################
            # load densified point clouds and associated time_indice
            data_path = lidarseg_path.replace('../../datasets/nuscene/v1.0-trainval/lidarseg/v1.0-trainval/','/net/pf-pc04/scratch3/nuscene/v1.0-trainval/lidarpoints/v1.0-trainval/').replace('.bin','.npy')
            data = np.load(data_path)
            time_indice, points = data[:,-1].astype(np.int), data[:,:3].astype(np.float64) 

            rec_path = lidarseg_path.replace('lidarseg','lidarrec_oracle').replace('bin','pth').replace('../../','../')
            data = torch.load(rec_path)
            bbox_tsfm, inst_labels = data['est_tsfm'].numpy(), data['inst_labels'].numpy()
            
            ###########################################
            # get fb_labels, sd_labels
            mask_num = bbox_num_pts > 0
            mask_name = np.array([True if ele in ['human','vehicle','animal'] else False for ele in bbox_names.tolist()])
            fb_bbox_mask = np.logical_and(mask_name, mask_num)
            fb_bbox_mask = np.concatenate([np.array([False]),fb_bbox_mask])

            velocity = np.linalg.norm(bbox_velocity, axis=1)
            dynamic_mask = velocity > 0.5
            dynamic_mask = np.concatenate([np.array([False]), dynamic_mask])

            fb_labels = fb_bbox_mask[inst_labels]
            sd_labels = dynamic_mask[inst_labels]

            import pdb
            pdb.set_trace()

            # check points in bbox part

            # count+=1
            # points = input_points[sel]
            # fb_label = fb_labels[sel]
            # sd_label = sd_labels[sel]
            # pcd_in = to_o3d_pcd(points)
            # pcd_fb = to_o3d_pcd(points[fb_label==0])
            # pcd_sd = to_o3d_pcd(points[sd_label==0])
            # multi_vis([pcd_in, pcd_fb, pcd_sd], ['input','bkgd','static'])

            # ego_motion_gt = torch.from_numpy(ego_motion_gt)
            # bbox_tsfm = torch.from_numpy(bbox_tsfm)
            # input_points = torch.from_numpy(input_points)
            # time_indice = torch.from_numpy(time_indice)
            # inst_labels = torch.from_numpy(inst_labels)
            # ego_compensated_points = ego_motion_compensation(input_points, time_indice, ego_motion_gt)
            # fully_reconstructed_points = reconstruct_sequence(ego_compensated_points, time_indice, inst_labels, bbox_tsfm,n_frames)
            # pcd_in = to_o3d_pcd(input_points)
            # pcd_rec = to_o3d_pcd(fully_reconstructed_points)
            # pcd_in.paint_uniform_color(get_blue())
            # pcd_rec.paint_uniform_color(get_blue())
            # multi_vis([pcd_in, pcd_rec], ['input','rec'])

    print(split, count)