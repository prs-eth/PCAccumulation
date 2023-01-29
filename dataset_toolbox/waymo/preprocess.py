import os,sys, time
from glob import glob
import numpy as np
from termcolor import colored
from utils import save_pkl, natural_key, load_pkl, to_o3d_pcd, vis_o3d, to_o3d_vec, multi_vis, get_blue, get_yellow
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from copy import deepcopy
import matplotlib.pyplot as plt
MY_CMAP = plt.cm.get_cmap('Set1')(np.arange(10))[:,:3]

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_decoder import extract_objects, extract_points
import tensorflow.compat.v1 as tf
from box_np_ops import center_to_corner_box3d, points_in_rbbox
from utils import setup_seed, makedirs
setup_seed(0)
import multiprocessing as mp

seqs = {
    'train':np.loadtxt('../reconstruction/configs/datasets/waymo/train.txt',dtype=str),
    'validation': np.loadtxt('../reconstruction/configs/datasets/waymo/validation.txt',dtype=str)
}
c_split = None


# https://github.com/waymo-research/waymo-open-dataset/blob/bbcd77fc503622a292f0928bfa455f190ca5946e/waymo_open_dataset/dataset.proto#L233

point_label = {
    1: 'vehicle',
    2: 'pedestrian',
    3: 'sign',
    4: 'cyclist'
}

def register_odometry_np(src, tsfm_src, tsfm_tgt):
    velo2cam = np.eye(4)

    M = (velo2cam @ tsfm_src.T @ np.linalg.inv(tsfm_tgt.T)@ np.linalg.inv(velo2cam)).T 

    R, t = M[:3,:3], M[:3,3][:,None]

    src = (R @ src.T + t).T
    return src

def get_split(ratio = 0.1):
    for split in['train','validation']:
        folder = f'/scratch3/waymo/{split}/*.tfrecord'
        sequences = glob(folder)
        print(colored('Found %d sequences' % len(sequences),'green'))
        np.random.shuffle(sequences)

        n_seqs = int(ratio * len(sequences))
        print(colored('Use %d sequences for %s' % (n_seqs,split),'green'))

        selected_seqs = sequences[:n_seqs]
        np.savetxt(f'configs/datasets/waymo/{split}.txt',np.array(selected_seqs), fmt='%s')

    

def mp_preprocess(c_split):
    """
    preprocess waymo data

    c_split = 'train'
    mp_preprocess(c_split)
    c_split = 'validation'
    mp_preprocess(c_split)
    """
    seqs_inds = np.arange(seqs[c_split].shape[0]).tolist()
    p = mp.Pool(processes=mp.cpu_count())
    p.map(process_seq, seqs_inds)
    p.close()
    p.join()


def check_data():
    # visually check data
    n_frames = 3
    interval = 1
    dist_threshold = 3
    voxel_size = 0.1
    velocity_threshold = 0.5  # threshold * 0.1 * interval * (n_frames - 1) > voxel_size 
    base = '/scratch3/waymo/customise/train'
    seqs = sorted(os.listdir(base),key=natural_key)

    for i in range(20):
        c_seq= seqs[np.random.randint(70)]
        lidars = sorted(glob(f'{base}/{c_seq}/lidar/*.npy'),key=natural_key)


        data_path = lidars[np.random.randint(10,len(lidars))]
        label_path = data_path.replace('lidar','label').replace('npy','pkl')
        final_pose = load_pkl(label_path)['veh_to_global']

        bin_name = int(os.path.basename(data_path).split('.')[0])
        dirname = os.path.dirname(data_path)

        # 1. get bbox information, and mapping to static/ dynamic; foreground/background
        meta_bbox = dict() # indexed by inst_name
        fb_mask_list = []
        bbox_name_list = []
        pose_list = []
        for c_idx in range(n_frames):
            c_bin_name = str(bin_name - c_idx * interval).zfill(4)+'.npy'
            c_data_path = os.path.join(dirname, c_bin_name)
            c_label_path =  c_data_path.replace('lidar','label').replace('npy','pkl')
            
            label_data = load_pkl(c_label_path)
            pose = label_data['veh_to_global']
            c_objects = label_data['objects']
            fb_mask = np.zeros(len(c_objects)+1)
            bbox_name = []
            for idx, eachbox in enumerate(c_objects):
                name = eachbox['name']
                label = eachbox['label']
                speed = np.linalg.norm(eachbox['global_speed'])
                fb_mask[idx+1] = int(label!=3)
                bbox_name.append(name)
                if(name not in meta_bbox):
                    meta_bbox[name]={
                        'label': label,
                        'speed': [speed]
                    }
                else:
                    meta_bbox[name]['speed'].append(speed)

            fb_mask_list.append(fb_mask)
            bbox_name_list.append(bbox_name)
            pose_list.append(pose)

        # assign dynamic/static label to each instance
        for key, value in meta_bbox.items():
            mean_speed = np.mean(value['speed'])
            max_speed = max(value['speed'])
            meta_bbox[key]['sd_label'] = int(np.logical_and(max_speed > velocity_threshold, value['label']!=3))

        # 2. get laser data
        time_index_list = []
        laser_data_list = []
        ground_height_list = []
        fb_label_list = []
        sd_label_list = []

        for c_idx in range(n_frames):
            c_bin_name = str(bin_name - c_idx * interval).zfill(4)+'.npy'
            c_data_path = os.path.join(dirname, c_bin_name)

            laser_data = np.load(c_data_path)
            pose = pose_list[c_idx]
            fb_mask = fb_mask_list[c_idx]
            bbox_name = bbox_name_list[c_idx]
            
            # 2.1 get ground height
            sel_ground_points = laser_data[:,3] > 0
            ground_points = laser_data[sel_ground_points,:3]
            dist = np.linalg.norm(ground_points[:,:2], axis=1)
            ground_height_list.append(ground_points[dist < dist_threshold, 2].mean())

            # 2.2 register point clouds to the last frame
            laser_data[:,:3] = register_odometry_np(laser_data[:,:3], pose, final_pose)
            laser_data_list.append(laser_data)
            
            time_index = np.ones(laser_data.shape[0]) * (n_frames - c_idx)
            time_index_list.append(time_index)

            # 2.3 get foreground/background label
            fb_label = fb_mask[(laser_data[:,-1]+1).astype(np.int)]
            fb_label_list.append(fb_label)

            # 2.4 get dynamic/static label
            sd_mask = np.zeros(len(bbox_name)+1)
            for idx,eachname in enumerate(bbox_name):
                sd_mask[idx+1] = meta_bbox[eachname]['sd_label']
            sd_label = sd_mask[(laser_data[:,-1]+1).astype(np.int)]
            sd_label_list.append(sd_label)

        laser_data = np.vstack(laser_data_list)
        time_indice = np.concatenate(time_index_list).astype(np.int)
        fb_label = np.concatenate(fb_label_list).astype(np.int)
        sd_label = np.concatenate(sd_label_list).astype(np.int)
        points = laser_data[:,:3]
        
        # 2. voxel_downsample
        coords = np.round(points / voxel_size)
        sel, _ = sparse_quantize(coords, return_index=True,return_invs=True)
        points, time_indice, sd_label, fb_label = points[sel],time_indice[sel], sd_label[sel], fb_label[sel] 
        laser_data = laser_data[sel]

        pcd_before = to_o3d_pcd(points)
        colors = MY_CMAP[time_indice]
        pcd_before.colors = to_o3d_vec(colors)

        # 3. remove ground points
        ground_height = max(ground_height_list) + 0.05
        sel_height = laser_data[:,2] > ground_height
        sel_sensor = laser_data[:,3] == 0
        sel = np.logical_and(sel_height, sel_sensor)
        points, time_indice, sd_label, fb_label = points[sel],time_indice[sel], sd_label[sel], fb_label[sel] 
        
        colors = MY_CMAP[time_indice]
        pcd_after = to_o3d_pcd(points)
        pcd_after.colors = to_o3d_vec(colors)

        # 4. color-code by foreground/background label
        colors[fb_label==0] = get_blue()
        colors[fb_label>0] = [0,0.8,0]
        colors[sd_label>0] = get_yellow()
        pcd_fb_label = deepcopy(pcd_after)
        pcd_fb_label.colors = to_o3d_vec(colors)

        # 5. color-code by dynamic/static label
        colors[sd_label>0] = get_yellow()
        colors[sd_label==0] = get_blue()
        pcd_sd_label = deepcopy(pcd_after)
        pcd_sd_label.colors = to_o3d_vec(colors)   

        # 4. assign dynamic objects
        multi_vis([pcd_before,pcd_after, pcd_fb_label, pcd_sd_label],[f'before: {len(pcd_before.points)} points',f'after: {len(pcd_after.points)} points','fb_gt',
        'sd_gt'], render=False)


def check_speed_info():
    speed_threshold = 0.1
    base = '/scratch3/waymo/customise'
    for split in ['train','validation']:
        c_base = os.path.join(base, split)
        seqs = os.listdir(c_base)
        for eachseq in seqs: # process each sequence
            meta_bbox = dict()
            c_folder = os.path.join(c_base, eachseq)
            labels = glob(c_folder+'/label/*.pkl')
            for eachlabel in labels:
                c_objects = load_pkl(eachlabel)['objects']
                for eachbox in c_objects:
                    name = eachbox['name']
                    label = eachbox['label']
                    speed = np.linalg.norm(eachbox['global_speed'])
                    if(name not in meta_bbox):
                        meta_bbox[name]={
                            'label': label,
                            'speed': [speed]
                        }
                    else:
                        meta_bbox[name]['speed'].append(speed)
            std_speed_list = []
            for key, value in meta_bbox.items():
                label = value['label']
                mean_speed = np.mean(value['speed'])
                std_speed = np.std(value['speed'])
                if mean_speed > speed_threshold:
                    std_speed_list.append(std_speed)
                #     plt.plot(value['speed'])
                #     plt.show()
            print(np.mean(std_speed_list))


def gen_test_anchors():
    base = '/scratch3/waymo/customise'
    split = 'validation'
    test_infos =[]
    for seq in range(20):
        seq = str(seq).zfill(3)
        lidars = sorted(glob(f'{base}/{split}/{seq}/lidar/*.npy'), key=natural_key)
        sel_lidars = lidars[10:] 
        sel_lidars = [ele.replace(base,'') for ele in sel_lidars]
        test_infos.extend(sel_lidars)
        np.savetxt(f'configs/datasets/waymo/meta_test.txt',test_infos, fmt = '%s')


if __name__=='__main__':
    # gen_test_anchors()
    # split ='train'
    # data = np.loadtxt(f'configs/datasets/waymo/meta_{split}.txt',dtype=str)
    # print(data.shape)
    # check_speed_info()
    c_split = 'train'
    mp_preprocess(c_split)