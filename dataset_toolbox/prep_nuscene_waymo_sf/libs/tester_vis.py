from copy import deepcopy
import torch, os, pickle
import numpy as np
import torch.nn as nn
from libs.utils import natural_key, save_pkl, to_o3d_pcd, vis_o3d, get_blue, get_yellow, to_o3d_vec, multi_vis, to_array, canonicalise_random_indice, makedirs, lighter, load_pkl, load_pkl, Logger, _EPS
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

Ground_height = {
    'waymo': 0.34,
    'nuscene': -1.54
}

class SegTrainer(BaseTrainer):
    def __init__(self, args):
        BaseTrainer.__init__(self,args)

        color_dict = load_pkl(args['path']['color_info'])
        self.color_dict = color_dict[99][1:]
        self.color_dict[0] = get_blue()
        self.color_dict = np.clip(self.color_dict, 0, 1)
        self.n_colors = self.color_dict.shape[0]
        self.n_frames = args['data']['n_frames']
        self.dataset = args['data']['dataset']
        
    
    def save_results_for_vis(self):
        # generate dataloader by scene names
        self.model.eval()
        
        model_name = self.save_dir.split('/')[1]     
        test_set = DATASETS[self.dataset](self.config, 'test', data_augmentation = False)
        dloader = torch.utils.data.DataLoader(test_set, 
                                batch_size=1, 
                                num_workers= 6, 
                                collate_fn=collate_fn,
                                sampler = DatasetSampler(test_set),
                                pin_memory=False,
                                drop_last=False)
        num_iter = len(test_set)
        c_loader_iter = dloader.__iter__()
        
        save_dir = f'results/supp/{model_name}'
        makedirs(save_dir)
        time_indice_list, fb_label_list, sd_label_list, relative_error_list, flow_error_list = [],[],[],[],[]
        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch
                
                # forward pass
                input_dict = c_loader_iter.next()
                for key, value in input_dict.items():
                    if(not isinstance(value, list)):
                        input_dict[key] = value.to(self.device)
                predictions = self.model(input_dict)
                self.loss(predictions, input_dict)
                
                # compute scene flow error
                input_points = input_dict['input_points']
                time_indice = input_dict['time_indice'][:,1].long()
                ego_motion_gt = input_dict['ego_motion_gt'].float()[0]
                inst_motion_gt = input_dict['inst_motion_gt'][0].to(self.device)
                inst_label_gt = input_dict['inst_labels'][:,0]
                fb_labels = input_dict['fb_labels'][:,0]
                sd_labels = input_dict['sd_labels'][:,0]
                sd_label_est = predictions['mos_est'].argmax(1)
                inst_label_est = predictions['inst_from_offset']
                scene_name, file_name = input_dict['data_path'][0].split('/')[-2], input_dict['data_path'][0].split('/')[-1].split('.')[0]      
                
                ratio = sd_labels.float().mean().item()
                if ratio > 0.1:   
                    ego_comp_gt = ego_motion_compensation(input_points, time_indice, ego_motion_gt)
                    full_rec_gt = reconstruct_sequence(ego_comp_gt, time_indice, inst_label_gt, inst_motion_gt, self.n_frames)
                    
                    ego_comp_est = predictions['transformed_points']
                    full_rec_est = predictions['rec_est']

                    est_flow = full_rec_est - input_points
                    gt_flow = full_rec_gt - input_points
                    error = est_flow - gt_flow
                    epe_per_point = torch.norm(error, p=2, dim=1)
                    gt_f_magnitude = torch.norm(gt_flow, p=2, dim=1)
                    relative_err = epe_per_point / (gt_f_magnitude + _EPS)
                    
                    sel = time_indice > 0
                    time_indice_list.extend(time_indice[sel].cpu().numpy().astype(np.int8))
                    fb_label_list.extend(fb_labels[sel].cpu().numpy().astype(np.bool))
                    sd_label_list.extend(sd_labels[sel].cpu().numpy().astype(np.bool))
                    relative_error_list.extend(relative_err[sel].cpu().numpy().astype(np.float16))
                    flow_error_list.extend(epe_per_point[sel].cpu().numpy().astype(np.float16))

                    idx = len(os.listdir(save_dir))
                    
                    if 'inst_pose_est' not in predictions.keys():
                        predictions['inst_pose_est'] = torch.eye(4).to(inst_motion_gt.device)
                    
                    static_epe = round(epe_per_point[sd_labels == 0].mean().item(), 4)
                    dynamic_epe = round(epe_per_point[sd_labels == 1].mean().item(), 4)         
                                    
                    inst_label_gt[sd_labels==0] = 0
                    colors_time = self.color_dict[time_indice.cpu().numpy()]
                    colors_sd_est = self.color_dict[sd_label_est.cpu().numpy()]
                    colors_inst_est = self.color_dict[inst_label_est.cpu().numpy() % self.color_dict.shape[0]]
                    colors_sd_gt = self.color_dict[sd_labels.cpu().numpy()]
                    colors_inst_gt = self.color_dict[inst_label_gt.cpu().numpy() % self.color_dict.shape[0]]

                    labels_est = inst_label_est + sd_label_est
                    labels_gt = inst_label_gt + sd_labels

                    labels_est = np.array(canonicalise_random_indice(labels_est.tolist()))
                    labels_gt = np.array(canonicalise_random_indice(labels_gt.tolist()))

                    colors_sd_inst_est = self.color_dict[labels_est % self.color_dict.shape[0]]
                    colors_sd_inst_gt = self.color_dict[labels_gt % self.color_dict.shape[0]]


                    relative_error_color = lighter(get_yellow(), 1 - relative_err[:,None].repeat(1,3).cpu().numpy() * 10)
                    epe_per_point = torch.clamp(epe_per_point, 0,1)
                    error_color = lighter(get_yellow(), 1 - epe_per_point[:,None].repeat(1,3).cpu().numpy())

                    pcd_rec_gt = to_o3d_pcd(full_rec_gt.cpu().numpy())
                    pcd_rec_gt.colors = to_o3d_vec(colors_sd_gt)

                    # color code by error ratio
                    pcd_rec_est = to_o3d_pcd(full_rec_est.cpu().numpy())
                    pcd_rec_est.colors = to_o3d_vec(error_color)

                    pcd_inst_est = to_o3d_pcd(ego_comp_est.cpu().numpy())
                    pcd_inst_est.colors = to_o3d_vec(colors_sd_inst_est)

                    pcd_inst_gt = to_o3d_pcd(ego_comp_gt.cpu().numpy())
                    pcd_inst_gt.colors = to_o3d_vec(colors_sd_inst_gt)
                    
                    
                    save_path = f"{save_dir}/{scene_name}_{file_name}.pkl"
                    self.logger.write(save_path+'\n')

                    #multi_vis([pcd_time, pcd_ego, pcd_inst, pcd_rec],['input','ego_motion_compensation','clustering','reconstruction'], render=False)
                    multi_vis([pcd_rec_gt, pcd_rec_est, pcd_inst_est, pcd_inst_gt],[f'gt rec{save_path}',f'pcd_ours: static_epe: {static_epe}, dynamic_epe: {dynamic_epe}', 'est inst', 'gt inst'], render=True)
                    with open(save_path, "wb") as f:
                        pickle.dump({
                            'input_points': input_dict['input_points'].cpu().numpy(),
                            'time_indice': time_indice.cpu().numpy(),
                            'fb_labels': fb_labels.cpu().numpy(),
                            'fb_est': predictions['fb_est_per_points'].cpu().numpy(),
                            'sd_labels': sd_labels.cpu().numpy(),
                            'sd_labels_est': sd_label_est.cpu().numpy(),
                            'inst_labels': inst_label_gt.cpu().numpy(),
                            'inst_labels_est': inst_label_est.cpu().numpy(),
                            'ego_motion_gt': ego_motion_gt.cpu().numpy(),
                            'ego_motion_est': predictions['ego_motion_est'].cpu().numpy()[0],
                            'inst_motion_gt': inst_motion_gt.cpu().numpy(),
                            'inst_motion_est': predictions['inst_pose_est'].cpu().numpy(),
                            'ego_only': ego_comp_gt.cpu().numpy(),
                            'full_rec': full_rec_gt.cpu().numpy(),
                            'offset_est': predictions['offset_est'].cpu().numpy(),
                            'offset_gt': predictions['offset_gt'].cpu().numpy(),
                            'flow_gt': gt_flow.cpu().numpy(),
                            'est_flow': est_flow.cpu().numpy()
                        }, f)

    def test(self):
        self.save_results_for_vis()
        