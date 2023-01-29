import sys, os, torch
import numpy as np
from libs.dataset import BaseDataset
from libs.utils import load_pkl, load_yaml, to_tensor, makedirs
from pyquaternion import Quaternion
from torchsparse import SparseTensor
from libs.bbox_utils import points_in_rbbox
from libs.register_utils import reconstruct_sequence, convert_rot_trans_to_tsfm, apply_tsfm
from copy import deepcopy
from scipy.spatial.transform import Rotation
from libs.dataset import DatasetSampler
from tqdm import tqdm
import glob
import random

def collate_fn(batch):
    pass

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



class NuSceneDataset(BaseDataset):
    """
    NuScene dataset for semantic segmentation and motion segmentation
    Here we only have labels for the keyframe
    nuScene ground height https://github.com/nutonomy/nuscenes-devkit/issues/161
    """
    def __init__(self,config,split, data_augmentation=True):
        BaseDataset.__init__(self, config, data_augmentation)

        self.split = split
        self.augmentation = data_augmentation
        info_path = f'configs/datasets/nuscene/metadata/{split}.pkl'
        self.nusc_infos = load_pkl(info_path)
        self.sem_learning_map = load_yaml(config['data']['label_map'])['learning_map']
        self.remove_close = config['data']['radius']

        self._make_folders()

        # determine whether or not we are running on Euler
        self.path_to_be_replaced = '/scratch3/nuScenes/v1.0-trainval'
        if(os.getcwd().find('/cluster/')==-1):
            self.path_to_replace = '../datasets/nuscene/v1.0-trainval'
        else:
            self.path_to_replace = config['path']['nuscene_dataset_euler']

        self.token_to_seg_path_mapping = load_pkl('../spv_SpatioTempo/dataset_configs/nuscenes/nuscene_token_to_seg_path.pkl')

    def _make_folders(self):
        scene_tokens = [ele['scene_token'] for ele in self.nusc_infos]
        unique_scene_tokens = list(set(scene_tokens))
        for scene_token in unique_scene_tokens:
            folder = f'/scratch3/cvpr2022/mini/nuscene/{self.split}/{scene_token}'
            makedirs(folder)
            

    def __len__(self):
        return len(self.nusc_infos)


    def _get_final_pose(self,info):
        l2e_r = info["lidar2ego_rotation"]
        l2e_t = np.array(info["lidar2ego_translation"])[:,None]
        e2g_r = info["ego2global_rotation"]
        e2g_t = np.array(info["ego2global_translation"])[:,None]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        rot = np.dot(e2g_r_mat, l2e_r_mat)
        trans = np.dot(e2g_r_mat, l2e_t) + e2g_t
        final_pose = np.eye(4)
        final_pose[:3,:3] = rot
        final_pose[:3,3] = trans.flatten()
        return final_pose


    def get_semseg_label(self, lidarseg_path):
        """
        Get labels for semantic segmentation
        """
        lidar_label = np.fromfile(lidarseg_path, dtype=np.uint8).reshape([-1])
        lidar_label = np.vectorize(self.sem_learning_map.__getitem__)(lidar_label).astype(np.uint8)
        return lidar_label


    def __getitem__(self, idx):
        info = self.nusc_infos[idx]

        ###########################################
        # load key frame and associated lidar labels
        lidar_path = info['lidar_path'].replace(self.path_to_be_replaced,self.path_to_replace)
        lidarseg_path = self.token_to_seg_path_mapping[info['token']]
        sem_labels = self.get_semseg_label(lidarseg_path.replace('../../','../'))

        lidar_points = read_nuscene_bin(lidar_path)
        ts = info['timestamp'] / 1e6
        ts_index = np.zeros((lidar_points.shape[0])).astype(np.int)
        bbox, bbox_velocity = info['gt_boxes'], info['gt_velocity']
        bbox_velocity = np.nan_to_num(bbox_velocity)
        bbox_num_pts = info['num_lidar_pts']
        bbox_names = info['gt_names']
        bbox_names = np.array([string_mapper[ele] for ele in bbox_names.tolist()])

        ###########################################
        # load densified point clouds and associated time_indice
        data_path = lidarseg_path.replace('../../datasets/nuscene/v1.0-trainval/lidarseg/v1.0-trainval/','/net/pf-pc04/scratch3/nuscene/v1.0-trainval/lidarpoints/v1.0-trainval/').replace('.bin','.npy')
        data = np.load(data_path)
        time_indice, points = data[:,-1].astype(np.int), data[:,:3].astype(np.float64) 
        if time_indice.max() != self.n_frames -1:
            print('found ... with less frames')
            return self.__getitem__(np.random.randint(0,self.__len__(),1)[0])

        rec_path = lidarseg_path.replace('lidarseg','lidarrec_oracle').replace('bin','pth').replace('../../','../')
        data = torch.load(rec_path)
        bbox_tsfm, inst_labels = data['est_tsfm'].numpy(), data['inst_labels'].numpy()
        
        ###########################################
        # get fb_labels, sd_labels
        mask_name = np.array([True if ele in ['human','vehicle','animal'] else False for ele in bbox_names.tolist()])
        fb_bbox_mask = mask_name
        fb_bbox_mask = np.concatenate([np.array([False]),fb_bbox_mask])

        velocity = np.linalg.norm(bbox_velocity, axis=1)
        dynamic_mask = velocity > self.speed_threshold
        dynamic_mask = np.concatenate([np.array([False]), dynamic_mask])

        fb_labels = fb_bbox_mask[inst_labels]
        sd_labels = dynamic_mask[inst_labels]

        n_bkgd = (fb_labels[time_indice==1] == 0).sum()
        if n_bkgd < 1024:
            print(n_bkgd, 'found few bkgd points here')

        ###########################################
        # load ground truth transformation matrix
        identity_matrix = np.eye(4)
        ego_motion_gt = [identity_matrix]
        sweep_info = info['sweeps']
        assert len(sweep_info) == self.n_frames - 1
        for idx, eachsweep in enumerate(sweep_info):
            rot, trans = eachsweep['sensor2lidar_rotation'], eachsweep['sensor2lidar_translation']
            c_tsfm = convert_rot_trans_to_tsfm(rot, trans)
            ego_motion_gt.append(c_tsfm)
        ego_motion_gt = np.array(ego_motion_gt)

        ###########################################
        # reverse gt ego-motion compensation
        for idx in range(self.n_frames):
            sel = time_indice == idx
            c_points = points[sel]
            c_points = (c_points - ego_motion_gt[idx,:3,3]) @ ego_motion_gt[idx,:3,:3]
            points[sel] = c_points
        
        # remove close points
        sel = np.logical_and(np.abs(points[:,0]) > self.remove_close, np.abs(points[:,1]) > self.remove_close)
        n_sample_pts = (time_indice == 0).sum()

        points, time_indice = points[sel], time_indice[sel]
        sd_labels, fb_labels, inst_labels = sd_labels[sel], fb_labels[sel], inst_labels[sel]
        sem_labels = sem_labels[sel[:n_sample_pts]]

        data = {
            'raw_points': points.astype(np.float32),
            'time_indice': time_indice.astype(np.int),
            'sd_labels': sd_labels.astype(np.int),
            'fb_labels': fb_labels.astype(np.int),
            'inst_labels': inst_labels.astype(np.int),
            'sem_labels': sem_labels.astype(np.int),
            'ego_motion_gt': ego_motion_gt.astype(np.float32),
            'bbox_tsfm': bbox_tsfm.astype(np.float32)
        }
        sample_token, scene_token = info['token'], info['scene_token']
        save_path = f'/scratch3/cvpr2022/mini/nuscene/{self.split}/{scene_token}/{sample_token}'
        np.savez_compressed(save_path, **data)

        #input_dict = self.prep_input(points, sd_labels, fb_labels, inst_labels, time_indice, ego_motion_gt, bbox_tsfm)

        return data



if __name__=='__main__':
    from libs.config import get_config
    config = get_config('configs/nuscene/nuscene.yaml')
    train_set = NuSceneDataset(config, 'train')
    val_set = NuSceneDataset(config, 'val')
    test_set = NuSceneDataset(config, 'test')
    dataloader = dict()
    dataloader['train'] = torch.utils.data.DataLoader(train_set, 
                                        batch_size=8, 
                                        num_workers=16,
                                        sampler = DatasetSampler(train_set),
                                        pin_memory=False,
                                        collate_fn=collate_fn,
                                        drop_last=False)
    dataloader['val'] = torch.utils.data.DataLoader(val_set, 
                                        batch_size=8, 
                                        num_workers=16,
                                        sampler = DatasetSampler(val_set),
                                        pin_memory=False,
                                        collate_fn=collate_fn,
                                        drop_last=False)
    dataloader['test'] = torch.utils.data.DataLoader(test_set, 
                                        batch_size=8, 
                                        num_workers=16,
                                        sampler = DatasetSampler(test_set),
                                        pin_memory=False,
                                        collate_fn=collate_fn,
                                        drop_last=False)

    for split in ['train','val','test']:
        num_iter = int(len(dataloader[split].dataset) // dataloader[split].batch_size)
        c_loader_iter = dataloader[split].__iter__()

        for idx in tqdm(range(num_iter)): # loop through this epoch
            input_dict = c_loader_iter.next()

    # for split in ['train','val','test']:
    #     files = glob.glob(f'/scratch3/cvpr2022/nuscene/{split}/*/*.npz')
    #     files = [ele.replace('/scratch3/cvpr2022','') for ele in files]
    #     np.savetxt(f'configs/datasets/nuscene/full_split/{split}_info.txt', files, fmt='%s')

    # # for our split, we use 240 for train, 60 for val, 60 for test
    # n_samples = {
    #     'train': 240,
    #     'val': 60,
    #     'test': 60
    # }
    # for split in ['train','val','test']:
    #     files = []
    #     folders = os.listdir(f'/scratch3/cvpr2022/nuscene/{split}')
    #     random.shuffle(folders)
    #     sampled_folders = folders[:n_samples[split]]
    #     for eachfolder in sampled_folders:
    #         files.extend(glob.glob(f'/scratch3/cvpr2022/nuscene/{split}/{eachfolder}/*.npz'))
    #     files = [ele.replace('/scratch3/cvpr2022','') for ele in files]
    #     np.savetxt(f'configs/datasets/nuscene/our_split/{split}_info.txt', files, fmt='%s')



