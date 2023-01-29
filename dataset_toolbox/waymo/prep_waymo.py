import sys, os
from glob import glob
import numpy as np
from libs.dataset import BaseDataset
from libs.utils import load_pkl, save_pkl, makedirs, natural_key
from libs.register_utils import register_odometry_np
from libs.simulate_tubes import InstanceObservations, get_max_rotation_along_z
from libs.config import get_config
import multiprocessing as mp
from tqdm import tqdm


point_label = {
    1: 'vehicle',
    2: 'pedestrian',
    3: 'sign',
    4: 'cyclist'
}
waymo = None

def get_annotations(bin_name, dirname):
    """
    Here we extract 
    1. pose information 
    2. foreground/background labels
    3. static/dynamic labels
    """
    meta_bbox = dict() # key = bbox_name
    bbox_name_list = []
    pose_list = [] # vehicle to global transformation per frame
    for c_idx in range(waymo.n_frames):
        # load bbox information
        c_bin_name = str(bin_name - c_idx * waymo.interval).zfill(4)+'.npy'
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
                    'sem_label': label -1,
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
        pose_list.append(pose)

    # assign static/dynamic label
    for key, value in meta_bbox.items():
        mean_speed = np.mean(value['speed'])
        max_speed = max(value['speed'])
        meta_bbox[key]['sd_label'] = int(np.logical_and(max_speed > waymo.speed_threshold, value['sem_label']!=2))
    
    pose_list = np.array(pose_list)
    return meta_bbox, bbox_name_list, pose_list


def get_item(idx):
    data_path = waymo.base + waymo.samples[idx]
    label_path = data_path.replace('lidar','label').replace('npy','pkl')
    obj_path = label_path.replace('label','obj')

    bin_name = int(os.path.basename(data_path).split('.')[0])
    dirname = os.path.dirname(data_path)
    final_pose = load_pkl(label_path)['veh_to_global']
    if(bin_name - waymo.n_frames * waymo.interval + 1 <0):
        return

    # 1. get annotations
    meta_bbox, bbox_name_list, pose_list = get_annotations(bin_name, dirname)
    meta_bbox['background'] = {
        'fb_label': 0,
        'sd_label': 0,
        'sem_label': 2,
        'bbox_index': 100000
    }

    # 1.1 filter frames if has too big rotation angle
    rot_along_z = get_max_rotation_along_z(pose_list)
    if rot_along_z > waymo.max_rot_along_z:
        return

    # 2. get laser data
    time_indice_list = []    # 1 means starting frame
    laser_data_list = []
    fb_label_list = []
    sd_label_list = []
    inst_label_list = []
    for c_idx in range(waymo.n_frames):
        c_bin_name = str(bin_name - c_idx * waymo.interval).zfill(4)+'.npy'
        c_data_path = os.path.join(dirname, c_bin_name)
        laser_data = np.load(c_data_path)
        bbox_indice = (laser_data[:,-1]+1).astype(np.int) # 0 means background
        pose = pose_list[c_idx]
        bbox_name = bbox_name_list[c_idx]
        bbox_name.insert(0,'background')
        bbox_name = np.array(bbox_name)
        bbox_names = bbox_name[bbox_indice]

        # 2.1 register point clouds to the last frame
        laser_data[:,:3] = register_odometry_np(laser_data[:,:3], pose, final_pose, 'waymo')
        laser_data_list.append(laser_data)
        
        # 2.2 per-point time_indice, this is consistent with bbox time indice
        time_indice = np.ones(laser_data.shape[0]) * (c_idx)
        time_indice_list.append(time_indice)

        # 2.3 get fb, sd, inst labels
        fb_label = [meta_bbox[ele]['fb_label'] for ele in bbox_names]
        sd_label = [meta_bbox[ele]['sd_label'] for ele in bbox_names]
        inst_label = [meta_bbox[ele]['bbox_index'] for ele in bbox_names]
        fb_label_list.append(fb_label)
        sd_label_list.append(sd_label)
        inst_label_list.append(inst_label)

    laser_data = np.vstack(laser_data_list)
    time_indice = np.concatenate(time_indice_list).astype(np.int)
    fb_label = np.concatenate(fb_label_list).astype(np.int)
    sd_label = np.concatenate(sd_label_list).astype(np.int)
    inst_label = np.concatenate(inst_label_list).astype(np.int)
    points = laser_data[:,:3]

    # remove bbox of class "sign"
    meta_bbox.pop('background',None)
    remove_list = [key for key in meta_bbox.keys() if meta_bbox[key]['fb_label']==0]
    for key in remove_list:
        meta_bbox.pop(key, None)

    # 3. extract instances 
    input_dict = {
        'points': points,
        'inst_label': inst_label,
        'time_indice': time_indice,
        'pose_list': pose_list,
        'meta_bbox': meta_bbox
    }

    inst_obs = InstanceObservations(input_dict, waymo.n_frames, min_points=10)
    simulated_tubes, real_tubes = inst_obs.simulated_tubes, inst_obs.real_tubes   # dictionary, has points and time_indice
    static_instances = inst_obs.static_instances

    if(len(simulated_tubes.keys()) * len(real_tubes.keys()) == 0):
        return 

    input_dict = {
        'simulated_tubes': simulated_tubes,
        'real_tubes': real_tubes,
        'pose_list': pose_list,
        'rot_along_z': rot_along_z
    }
    save_pkl(input_dict, obj_path)


class WaymoDataset:
    """
    bbox[:,-1] is in range [-pi, pi]
    In this dataloader, we prepare instances
    
    """

    def __init__(self,config,split):
        super().__init__()       
        # get train/val/test meta_info
        self.split = split
        self.samples = np.loadtxt(f'configs/waymo/meta_{split}.txt',dtype=str)
        self.base = config['path']['waymo_dataset']

        self.speed_threshold = config['data']['speed_threshold']
        self.max_rot_along_z = config['data']['max_rot_along_z']
        self.n_frames = config['data']['n_frames']
        self.interval = config['data']['interval']


def generate_full_metadata():
    base = '/scratch3/waymo/customise'
    for split in ['training','validation']:
        seqs = os.listdir(os.path.join(base, split))
        anchors = []
        for eachseq in seqs:
            lidars = sorted(glob(f'{base}/{split}/{eachseq}/lidar/*.npy'), key=natural_key)
            sel_lidars = lidars[5::5] 
            sel_lidars = [ele.replace(base,'') for ele in sel_lidars]
            anchors.extend(sel_lidars)
        np.savetxt(f'configs/waymo/meta_{split}.txt',anchors, fmt = '%s')


def make_folders():
    base = '/scratch3/waymo/customise'
    for split in ['training','validation']:
        seqs = os.listdir(os.path.join(base, split))
        for eachseq in seqs:
            obj_dir = os.path.join(base, split, eachseq,'obj')
            makedirs(obj_dir)


if __name__=='__main__':
    # generate_full_metadata()
    # make_folders()
    # config = get_config('configs/waymo/tpointnet.yaml')
    # for split in ['train','val']:
    #     waymo = WaymoDataset(config, split)
    #     inds = np.arange(len(waymo.samples)).tolist()
    #     p = mp.Pool(processes=mp.cpu_count())
    #     p.map(get_item, inds)
    #     p.close()
    #     p.join()
    source_folder = '/scratch3/waymo/customise'
    target_folder = '/scratch2/shengyu/datasets/waymo_obj'
    for split in ['training','validation']:
        base = os.path.join(source_folder, split)
        seqs = os.listdir(base)
        for eachseq in tqdm(seqs):
            target_path = os.path.join(target_folder, split,eachseq)
            makedirs(target_path)
            obj_path = os.path.join(base, eachseq, 'obj')
            os.system(f'cp -r {obj_path} {target_path}')
            