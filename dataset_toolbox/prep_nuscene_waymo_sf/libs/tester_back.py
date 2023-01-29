from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
from libs.utils import save_pkl, to_o3d_pcd, vis_o3d, get_blue, get_yellow, to_o3d_vec, multi_vis, to_array, canonicalise_random_indice, makedirs, lighter, load_pkl, load_pkl, Logger, _EPS
import open3d as o3d
from libs.loss import compute_iou
from libs.metrics import init_stats_meter, update_stats_meter, compute_mean_iou_recall_precision
from libs.sf_eval_utils import  SF_Evaluator, compute_flow_error_torch, compute_sf_metrics_torch, display_from_stats_meter

from tqdm import tqdm
from libs.trainer import BaseTrainer
import matplotlib.pyplot as plt
from libs.register_utils import ego_motion_compensation, reconstruct_sequence
from libs.dataset import DatasetSampler, NuSceneDataset, WaymoDataset
from libs.dataloader import collate_fn
DATASETS = {
    'nuscene': NuSceneDataset,
    'waymo': WaymoDataset
}



def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent


class SegTrainer(BaseTrainer):
    def __init__(self, args):
        BaseTrainer.__init__(self,args)

        color_dict = load_pkl(args['path']['color_info'])
        self.color_dict = color_dict[99][1:]
        self.color_dict[0] = get_blue()
        self.color_dict = np.clip(self.color_dict, 0, 1)
        self.n_colors = self.color_dict.shape[0]
        self.n_frames = args['data']['n_frames']
        self.sf_evaluator = SF_Evaluator(self.n_frames, args['save_dir'])
        self.dataset = args['data']['dataset']

            
            

                

    def check_speed(self):
        split = 'test'
        print(f'Evaluate the pipeline on {split} dataset')
        num_iter = int(len(self.loader[split].dataset) // self.loader[split].batch_size)
        c_loader_iter = self.loader[split].__iter__()
        self.model.eval()
        
        time_dict = dict()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch
                input_dict = c_loader_iter.next()

                for key, value in input_dict.items():
                    if(not isinstance(value, list)):
                        input_dict[key] = value.to(self.device)

                predictions = self.model(input_dict)
                
                if idx == 0:
                    for key, value in predictions['time'].items():
                        time_dict[key] = [value]
                else:
                    for key, value in predictions['time'].items():
                        time_dict[key].append(value)
        
        for key, value in time_dict.items():
            print(key, np.mean(value))


    def test(self):
        # self.visualise_bev()
        # self.evaluate_scene_flow(vis=True)
        self.check_speed()