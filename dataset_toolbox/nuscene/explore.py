from nuscenes import NuScenes
import numpy as np
import pdb, time, os, sys, pickle
from glob import glob
from pathlib import Path
import open3d as o3d
from nuscene import NuScenesDataset
from libs.utils import to_o3d_pcd, get_blue, get_yellow, multi_vis, vis_o3d, to_o3d_vec
from libs.bbox_utils import center_to_corner_box3d, points_in_rbbox, corners_to_lines
from libs.remove_ground import get_non_ground

from libs.utils import load_pkl
color_dict = load_pkl('assets/distinct_colors.pkl')[99][1:]
color_dict = np.clip(color_dict, 0,1)

"""
Reorganise data follow this: 
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_lidarseg_panoptic_tutorial.ipynb

samples	-	Sensor data for keyframes (annotated images).
sweeps  -   Sensor data for intermediate frames (unannotated images).
"""
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


NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}




def explore_kit():
    version = 'v1.0-mini'
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
    

    # idx = np.random.randint(0,n_samples,1)[0]
    # my_sample = nusc.sample[idx]
    # idx = np.random.randint(0,n_scenes,1)[0]
    # my_scene = nusc.scene[idx]
    # nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count')


    # render lidar to BEV image
    # sample_data_token = my_sample['data']['LIDAR_TOP']
    # nusc.render_sample_data(sample_data_token,with_anns=False,show_lidarseg=True,show_lidarseg_legend=True)

    # render all sensor data
    # nusc.render_sample(my_sample['token'],show_lidarseg=True,filter_lidarseg_labels=[22, 23])

    # nusc.render_scene_channel_lidarseg(my_scene['token'], 
    #                                'CAM_BACK', 
    #                                verbose=True, 
    #                                dpi=100,
    #                                imsize=(1280, 720))



if __name__=='__main__':
    samples = os.listdir('/scratch3/nuScenes/compressed/v1.0-trainval01_blobs_lidar/samples/LIDAR_TOP')
    key = os.listdir(('/scratch3/nuScenes/v1.0-trainval/samples/LIDAR_TOP'))
    for eachsample in samples:
        assert eachsample in key

    samples = os.listdir('/scratch3/nuScenes/compressed/v1.0-trainval01_blobs_lidar/sweeps/LIDAR_TOP')
    key = os.listdir(('/scratch3/nuScenes/v1.0-trainval/sweeps/LIDAR_TOP'))
    for eachsample in samples:
        assert eachsample in key