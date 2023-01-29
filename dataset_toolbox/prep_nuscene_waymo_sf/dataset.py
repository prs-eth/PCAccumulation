import os, random
import numpy as np
from torch.utils.data import Dataset, Sampler
from scipy.spatial.transform import Rotation
from libs.register_utils import apply_tsfm
from libs.voxel_generator import Voxelization
from torchsparse.utils import sparse_quantize
from libs.utils import Logger

class DatasetSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indice = np.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        np.random.shuffle(self.indice)
        return iter(self.indice)


class BaseDataset(Dataset):
    """
    Dataset base class for NuScene and Waymo
    We return:  
        raw_points (m x 3):                  input_point clouds without voxelisation
        lidar_list ([n_frames]):             a list of voxelised sparse tensor
        ego_motion_gt ([n_frames x 4 x 4]):  ground truth ego motion, anchor frame undergoes zero motion
        labels ([n, 1]):                     static/dynamic labels
        fb_labes ([n, 1]):                   foreground/background labels
        label_masks ([n, 1]):                boolean, mask used to compute motion segmentation loss
        inst_labels ([n, 1])                 instance labels, 0 means static background
        inverse_map:([m, 1]):                inversion from voxelised points to raw points
        length_raw (integer):                m, number of raw points
        length_voxel (integer):              n, number of points after voxelisation
        bbox_tsfm ([K, n_frames, 4, 4])      transformation parameters of foreground objects, points in the anchor frame undergo zero motion
    To recover ground truth reconstruction, we compute:
    1. ego-motion compensated points by using ego_motion_gt
    2. moving-object-motion compensated points by using bbox_tsfm over ego-motion-compensated points
    """
    def __init__(self,config,split,data_augmentation=True, scene_name = None):
        super(BaseDataset,self).__init__()

        # data augmentation configs
        self.augmentation = data_augmentation
        self.augment_noise = config['data_aug']['augment_noise']
        self.augment_shift_range = config['data_aug']['augment_shift_range']
        self.augment_scale_min = config['data_aug']['augment_scale_min']
        self.augment_scale_max = config['data_aug']['augment_scale_max']
        self.rot_aug = config['data_aug']['rot_aug']
        
        # voxeliser
        self.voxeliser = Voxelization(config['voxel_generator'])
        self.n_frames = config['data']['n_frames']
        self.crop_xy = config['voxel_generator']['crop_range'][0]
        self.crop_z_min = config['voxel_generator']['crop_range'][1]
        self.crop_z_max = config['voxel_generator']['crop_range'][2]
        self.pre_voxel_size = config['data']['pre_voxel_size']
        self.apply_pre_voxel_downsample = config['data']['pre_voxel_downsample']
        
        self.remove_ground = config['data']['remove_ground']
        self.ground_height = config['data']['ground_height'] + config['data']['ground_slack']
        
        self.logger = Logger(config['save_dir'])

        # meta data
        if os.path.exists(config['path']['dataset_base_local']):
            self.base = config['path']['dataset_base_local']
        elif os.path.exists(config['path']['dataset_base_server']):
            self.base = config['path']['dataset_base_server']
        else:
            raise Exception('Unable to find datatset')
    
        self.infos = np.loadtxt(self.DATA_FILES[split], dtype= str)
        random.shuffle(self.infos)
        
        # filter out the scenes 
        if scene_name is not None:
            all_scene_names = [ele.split('/')[3] for ele in self.infos.tolist()]
            sel = [True if ele == scene_name else False for ele in all_scene_names]
            sel = np.array(sel)
            self.infos = self.infos[sel]
        
        n_samples = len(self.infos)
        scenes = [ele.split('/')[3] for ele in self.infos]
        n_unique_scenes = len(list(set(scenes))) 
        
        self.mode = config['misc']['mode']
        self.logger.write(f'We have {n_samples} samples from {n_unique_scenes} scenes for {split} split\n')

    def __len__(self):
        return len(self.infos)


    def apply_data_augmentation(self, points):
        """
        Apply data augmentation
        """
        # add gaussian noise
        points+= (np.random.rand(points.shape[0],3) - 0.5) * self.augment_noise
        
        # scale the point cloud
        scale_factor = np.random.uniform(self.augment_scale_min, self.augment_scale_max)
        points = points * scale_factor

        return points


    def _sample_random_tsfm(self):
        """
        We sample a random transformation [4, 4]
        """
        euler=[0,0,np.random.uniform(0,np.pi*self.rot_aug)] # anglez, angley, anglex
        rot= Rotation.from_euler('xyz', euler).as_matrix()
        shift = [np.random.uniform(-self.augment_shift_range, self.augment_shift_range), np.random.uniform(-self.augment_shift_range, self.augment_shift_range), 0]
        tsfm = np.eye(4)
        tsfm[:3,:3] = rot
        tsfm[:3,3] = np.array(shift)
        return tsfm

    def update_transformation_after_data_augmentation(self, aug_tsfm, ego_motion, inst_motion):
        """
        X_0 =  T_ego @ X_1
        T' @ X_0 = T' @ (T_ego @ T'^-1) @ (T' @ X_1) 
        T'_ego = T' @ T_ego @ T'^-1
        
        For anchor frame, it remains to undergo zero motion, for both static and dynamic parts
        Input:
            aug_tsfm:       [4, 4]
            ego_motion:     [n_frames, 4, 4]
            inst_motion:    [K, n_frames, 4, 4]
        """
        c_aug_tsfm = aug_tsfm[None].repeat(self.n_frames, 0)
        ego_motion = c_aug_tsfm @ ego_motion @ np.linalg.inv(c_aug_tsfm)

        inst_motion = inst_motion.reshape(-1,4,4)
        c_aug_tsfm = aug_tsfm[None].repeat(inst_motion.shape[0], 0)
        inst_motion = c_aug_tsfm @ inst_motion @ np.linalg.inv(c_aug_tsfm)
        inst_motion = inst_motion.reshape(-1,self.n_frames, 4, 4)

        return ego_motion, inst_motion
    
    
    def voxel_downsample(self, points):
        coords = np.round(points / self.pre_voxel_size)
        sel, inverse_map = sparse_quantize(coords,return_index=True,return_invs=True)

        return coords, sel, inverse_map
        
    def prep_input(self, raw_points, sd_labels, fb_labels, inst_labels, time_indice, ego_motion_gt, inst_motion_gt):
        """
        Inut:
            raw_points:     [m, 3]              points before ego-motion compensation
            sd_labels:      [m]              
            fb_labels:      [m]
            inst_labels:    [m]
            time_indice:    [m]
            ego_motion_gt:  [n_frames, 4, 4]
            inst_motion_gt: [k, n_frames, 4, 4]
        """
        assert raw_points.shape[0] == sd_labels.shape[0] == fb_labels.shape[0]==inst_labels.shape[0]==time_indice.shape[0]
        assert ego_motion_gt.shape[0] == self.n_frames
        assert inst_motion_gt.shape[1] == self.n_frames
        
        # 1. apply data augmentation
        if self.augmentation:
            random_tsfm = self._sample_random_tsfm()
            raw_points = apply_tsfm(raw_points, random_tsfm)
            raw_points = self.apply_data_augmentation(raw_points)
            ego_motion_gt, inst_motion_gt = self.update_transformation_after_data_augmentation(random_tsfm, ego_motion_gt, inst_motion_gt)
            
        # 2. crop the scene
        sel_xy = np.logical_and(np.abs(raw_points[:,0]) < self.crop_xy, np.abs(raw_points[:,1]) < self.crop_xy)
        sel_z = np.logical_and(raw_points[:,2] < self.crop_z_max, raw_points[:,2] > self.crop_z_min)
        sel_scene = np.logical_and(sel_xy, sel_z)
               
        raw_points, time_indice = raw_points[sel_scene], time_indice[sel_scene]
        sd_labels, fb_labels = sd_labels[sel_scene], fb_labels[sel_scene]
        inst_labels = inst_labels[sel_scene]
        
        # remove ground points
        sel_non_ground = raw_points[:,2] > self.ground_height 
        if self.remove_ground:
            raw_points, time_indice = raw_points[sel_non_ground], time_indice[sel_non_ground]
            sd_labels, fb_labels = sd_labels[sel_non_ground], fb_labels[sel_non_ground]
            inst_labels = inst_labels[sel_non_ground]

        # 3. remove frames 
        n_used_sweeps = self.voxeliser.n_sweeps
        sel = time_indice<n_used_sweeps
        raw_points, time_indice = raw_points[sel], time_indice[sel]
        sd_labels, fb_labels, inst_labels = sd_labels[sel], fb_labels[sel], inst_labels[sel]
        ego_motion_gt = ego_motion_gt[:n_used_sweeps]
        
        
        # 4. voxel-downsample to deal with the density imbalance
        if self.apply_pre_voxel_downsample:
            down_sampled_points_list = []
            inverse_map_list = []
            sd_label_list, inst_label_list, time_indice_list, fb_label_list = [], [], [],[]
            n_accumulated = 0
            for time_idx in range(self.voxeliser.n_sweeps):
                sel = time_indice == time_idx
                c_points = raw_points[sel]

                _, sub_sel, inverse_map = self.voxel_downsample(c_points)
                inverse_map += n_accumulated
                n_accumulated += sub_sel.shape[0]
                
                down_sampled_points_list.append(raw_points[sel][sub_sel])
                inverse_map_list.append(inverse_map)
                sd_label_list.append(sd_labels[sel][sub_sel])
                inst_label_list.append(inst_labels[sel][sub_sel])
                time_indice_list.append(np.ones_like(sub_sel) * time_idx)
                fb_label_list.append(fb_labels[sel][sub_sel])
            
            down_sampled_points = np.concatenate(down_sampled_points_list)
            sd_labels = np.concatenate(sd_label_list)
            fb_labels = np.concatenate(fb_label_list)
            inst_labels = np.concatenate(inst_label_list)   
            inverse_map = np.concatenate(inverse_map_list)
            time_indice = np.concatenate(time_indice_list).astype(np.int)
        else:
            inverse_map = np.arange(raw_points.shape[0])
            down_sampled_points = raw_points

        # 5. convert to point pillars
        points = np.concatenate((down_sampled_points, time_indice[:,None]), axis=1).astype(np.float32)
        voxels = self.voxeliser(points)
        num_points = np.array([down_sampled_points.shape[0]], dtype=np.int64)
        num_points_raw = np.array([raw_points.shape[0]], dtype=np.int64)
        
        data = {
            'input_points': down_sampled_points,
            'num_points': num_points, 
            'time_indice': time_indice[:,None],
            'sd_labels': sd_labels[:,None],
            'inst_labels': inst_labels[:,None],
            'ego_motion_gt': ego_motion_gt,
            'inst_motion_gt': inst_motion_gt,
            'fb_labels': fb_labels[:,None]
        }
        
        for key, value in voxels.items():
            data[key] = value
        
        return data


    def __getitem__(self, idx):
        data_path = self.base + self.infos[idx]
        data = np.load(data_path, allow_pickle=True)

        raw_points, time_indice = data['raw_points'], data['time_indice']
        sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
        inst_labels, sem_labels = data['inst_labels'], data['sem_labels']
        ego_motion_gt, inst_motion_gt = data['ego_motion_gt'], data['bbox_tsfm']

        input_dict = self.prep_input(raw_points, sd_labels, fb_labels, inst_labels, time_indice, ego_motion_gt, inst_motion_gt)
        
        if (input_dict['point_to_voxel_map'].min() != 0) or (input_dict['point_to_voxel_map'].max() +1 != input_dict['num_voxels']):
            self.logger.write(f'{data_path} is not working\n')
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        if self.mode == 'test':
            input_dict['data_path'] = data_path

        return input_dict

class NuSceneDataset(BaseDataset):
    DATA_FILES = {
        'train': '../assets/configs/datasets/nuscene/full_split/train_info.txt',
        'val': '../assets/configs/datasets/nuscene/full_split/val_info.txt',
        'test': '../assets/configs/datasets/nuscene/full_split/test_info.txt',
    } 

class WaymoDataset(BaseDataset):
    DATA_FILES = {
        'train': '../assets/configs/datasets/waymo/full_split/train_info.txt',
        'val': '../assets/configs/datasets/waymo/full_split/val_info.txt',
        'test': '../assets/configs/datasets/waymo/full_split/test_info.txt',
    } 
    
if __name__ == '__main__':
    info = np.loadtxt('configs/datasets/waymo/full_split/test_info.txt',dtype=str)
    base = '/net/pf-pc04/scratch3/cvpr2022/mini'
    import torch
    for idx in range(0,1000,5):
        path = base + info[idx]
        data = np.load(path)
        raw_points, time_indice = data['raw_points'], data['time_indice']
        sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
        inst_labels, sem_labels = data['inst_labels'], data['sem_labels']
        ego_motion_gt, inst_motion_gt = data['ego_motion_gt'], data['bbox_tsfm']
        
        # ego_motion_gt = torch.from_numpy(ego_motion_gt)
        # inst_motion_gt = torch.from_numpy(inst_motion_gt)[:,0]
        # print(torch.abs(ego_motion_gt[0] - torch.eye(4)).sum())
        # print(torch.abs(inst_motion_gt[0] - torch.eye(4)[None].repeat(inst_motion_gt.size(0), 1,1)).sum())
        
        import pdb
        pdb.set_trace()