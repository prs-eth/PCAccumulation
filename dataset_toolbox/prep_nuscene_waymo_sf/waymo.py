import sys, os, torch, glob
import numpy as np
from libs.dataset import BaseDataset
from libs.utils import load_pkl, save_pkl, makedirs, to_tensor, to_array, natural_key
from libs.register_utils import register_odometry_np, reconstruct_sequence, kabsch_transformation_estimation, convert_rot_trans_to_tsfm, apply_tsfm
from libs.bbox_utils import center_to_corner_box3d
from torchsparse import SparseTensor
from scipy.spatial.transform import Rotation
from libs.dataset import DatasetSampler
from tqdm import tqdm

class WaymoDataset(torch.utils.data.Dataset):
    """
    bbox[:,-1] is in range [-pi, pi]
    """
    def __init__(self,config):
        # get train/val/test meta_info
        self.interval = config['data']['interval']
        self.n_frames = config['data']['n_frames']
        self.speed_threshold = config['data']['speed_threshold']

        # get scenes 
        self.base = '/scratch3/waymo/our_format/validation'
        self.scenes = sorted(os.listdir(self.base), key = natural_key)
        self._make_folder()

        # get samples 
        self._make_samples()

    
    def _make_samples(self):
        target_path = 'waymo_generalisation_samples.txt'
        self.samples = []
        if not os.path.exists(target_path):
            for eachscene in self.scenes:
                files = sorted(glob.glob(self.base + '/' + eachscene + '/lidar/*.npy'), key=natural_key)
                sampled_keys = files[::self.n_frames]
                self.samples.extend(sampled_keys)
            self.samples = np.array(self.samples)
            np.savetxt(target_path, self.samples,fmt = "%s",delimiter='\n')
        else:
            self.samples = np.loadtxt(target_path,dtype = str)


    def _make_folder(self):
        for eachscene in self.scenes:
            target_dir = os.path.join('/scratch3/eccv2022/compressed/waymo_generalisation/test', eachscene)
            makedirs(target_dir)

    def _get_annotations(self, bin_name, dirname, anchor_pose):
        """
        Here we extract 
        1. ego-motion, here the anchor frame undergoes zero motion
        2. foreground/background labels
        3. static/dynamic labels
        """
        meta_bbox = dict() # key = bbox_name
        bbox_name_list = []
        ego_motion_gt_list = [] # vehicle to global transformation per frame
        for c_idx in range(self.n_frames):
            # load bbox information
            c_bin_name = str(bin_name - c_idx * self.interval).zfill(4)+'.npy'
            c_data_path = os.path.join(dirname, c_bin_name)
            c_label_path =  c_data_path.replace('lidar','label').replace('npy','pkl')
            label_data = load_pkl(c_label_path)
            pose = label_data['veh_to_global']
            c_objects = label_data['objects']
            bbox_name = []
            for idx, eachbox in enumerate(c_objects):
                bbox = eachbox['box']
                name = eachbox['name']
                label = eachbox['label']
                speed = np.linalg.norm(eachbox['global_speed'])
                bbox_name.append(name)

                if(name not in meta_bbox):
                    n_bbox = len(meta_bbox.keys())
                    meta_bbox[name]={
                        'bbox_index': n_bbox,
                        'sem_label': label - 1,
                        'fb_label': int(label!=3),
                        'speed': [speed],
                        'bbox': [bbox],
                        'time_indice': [c_idx]
                    }
                else:
                    meta_bbox[name]['speed'].append(speed)
                    meta_bbox[name]['bbox'].append(bbox)
                    meta_bbox[name]['time_indice'].append(c_idx)
                
            bbox_name_list.append(bbox_name)
            relative_pose = np.linalg.inv(anchor_pose) @ pose
            ego_motion_gt_list.append(relative_pose)

        ego_motion_gt = np.array(ego_motion_gt_list)
        return meta_bbox, bbox_name_list, ego_motion_gt
    

    def _get_bbox_tsfm(self, meta_bbox, ego_motion_gt):
        # assign static/dynamic label
        for key, value in meta_bbox.items():
            mean_speed = np.mean(value['speed'])
            max_speed = max(value['speed'])
            meta_bbox[key]['sd_label'] = int(np.logical_and(max_speed > self.speed_threshold, value['sem_label']!=2))

        # get transformation matrix for each instance
        zero_motion = np.eye(4)[None].repeat(self.n_frames, 0)
        bbox_tsfm = [zero_motion]
        for key, value in meta_bbox.items():
            if value['sd_label'] == 0:
                bbox_tsfm.append(zero_motion)
            else:
                bbox = np.array(value['bbox'])  #[N, 7]
                corners_idx = value['time_indice'] #[N]
                corners = center_to_corner_box3d(bbox[:,:3], bbox[:,3:6],-bbox[:,-1])  # [N, 8, 3]
                anchor_corners = corners[0]
                anchor_idx = corners_idx[0]
                anchor_corners = apply_tsfm(anchor_corners, ego_motion_gt[anchor_idx])
                bbox_tsfm_list = []
                for idx in range(self.n_frames):
                    if(idx in corners_idx):
                        c_corners = corners[corners_idx.index(idx)]  #[8,3]
                        c_corners = apply_tsfm(c_corners, ego_motion_gt[idx]).astype(np.float32)

                        rotation_matrix, translation_matrix, res, _ = kabsch_transformation_estimation(to_tensor(c_corners).float()[None],to_tensor(anchor_corners).float()[None])
                        c_tsfm = convert_rot_trans_to_tsfm(to_array(rotation_matrix[0]), to_array(translation_matrix[0]))
                        bbox_tsfm_list.append(c_tsfm)
                    else:
                        bbox_tsfm_list.append(np.eye(4))
                bbox_tsfm.append(np.array(bbox_tsfm_list))
        
        bbox_tsfm = np.array(bbox_tsfm)
        return bbox_tsfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path = self.samples[idx]
        label_path = data_path.replace('lidar','label').replace('npy','pkl')
        final_pose = load_pkl(label_path)['veh_to_global']

        bin_name = int(os.path.basename(data_path).split('.')[0])
        dirname = os.path.dirname(data_path)
        if(bin_name - self.n_frames * self.interval + 1 <0):
            return self.__getitem__(np.random.randint(0,self.__len__(),1)[0])

        # 1. get annotations
        meta_bbox, bbox_name_list, ego_motion_gt = self._get_annotations(bin_name, dirname, final_pose)
        bbox_tsfm = self._get_bbox_tsfm(meta_bbox, ego_motion_gt)
        meta_bbox['background'] = {
            'fb_label': 0,
            'sd_label': 0,
            'bbox_index': 100000,
            'sem_label': 2
        }

        # 2. get time indice, instance label, semantic label, and mos label
        time_indice_list = []    
        laser_data_list = []
        fb_label_list = []
        sd_label_list = []
        sem_label_list = []
        inst_label_list = []
        for c_idx in range(self.n_frames):
            c_bin_name = str(bin_name - c_idx * self.interval).zfill(4)+'.npy'
            c_data_path = os.path.join(dirname, c_bin_name)
            laser_data = np.load(c_data_path)
            bbox_indice = (laser_data[:,-1]+1).astype(np.int) # 0 means background
            pose = ego_motion_gt[c_idx]
            bbox_name = bbox_name_list[c_idx]
            bbox_name.insert(0,'background')
            bbox_name = np.array(bbox_name)

            # 1. select points from a specific sensor
            sel = laser_data[:,3] == 0
            laser_data, bbox_indice = laser_data[sel], bbox_indice[sel]
            bbox_names = bbox_name[bbox_indice]
            laser_data_list.append(laser_data)
            
            # 2 per-point time_indice, this is consistent with bbox time indice
            time_indice = np.ones(laser_data.shape[0]) * (c_idx)
            time_indice_list.append(time_indice)

            # 3 get fb, sd, inst labels
            fb_label = np.array([meta_bbox[ele]['fb_label'] for ele in bbox_names])
            sd_label = np.array([meta_bbox[ele]['sd_label'] for ele in bbox_names])
            sem_label = np.array([meta_bbox[ele]['sem_label'] for ele in bbox_names])
            inst_label = np.array([meta_bbox[ele]['bbox_index'] for ele in bbox_names])

            fb_label_list.append(fb_label)
            sd_label_list.append(sd_label)
            sem_label_list.append(sem_label)
            inst_label_list.append(inst_label)

        laser_data = np.vstack(laser_data_list)
        points = laser_data[:,:3]
        time_indice = np.concatenate(time_indice_list).astype(np.int)
        sd_labels = np.concatenate(sd_label_list).astype(np.int)
        fb_labels = np.concatenate(fb_label_list).astype(np.int)
        sem_labels = np.concatenate(sem_label_list).astype(np.int)
        inst_labels = np.concatenate(inst_label_list).astype(np.int)
        inst_labels +=1 
        inst_labels[inst_labels == 100001] = 0

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

        scene_token = data_path.split('/')[5]
        sample_token = data_path.split('/')[-1].split('.')[0]
        save_path = f'/scratch3/eccv2022/compressed/waymo_generalisation/test/{scene_token}/{sample_token}'
        np.savez_compressed(save_path, **data)
        #input_dict = self.prep_input(points, sd_labels, fb_labels, inst_labels, time_indice, ego_motion_gt, bbox_tsfm)
        return data

if __name__=='__main__':
    from libs.config import get_config
    config = get_config('configs/waymo.yaml')
    test_set = WaymoDataset(config)
    dataloader = dict()
    dataloader['test'] = torch.utils.data.DataLoader(test_set, 
                                        batch_size=1, 
                                        num_workers=config['train']['num_workers'],
                                        sampler = DatasetSampler(test_set),
                                        pin_memory=False,
                                        drop_last=False)
    split = 'test'
    num_iter = int(len(dataloader[split].dataset) // dataloader[split].batch_size)
    c_loader_iter = dataloader[split].__iter__()

    for idx in tqdm(range(num_iter)): # loop through this epoch
        input_dict = c_loader_iter.next()