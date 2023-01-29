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
        
    def visualise_bev(self):
        split = 'test'
        print(f'Evaluate the pipeline on {split} dataset')
        num_iter = int(len(self.loader[split].dataset) // self.loader[split].batch_size)
        c_loader_iter = self.loader[split].__iter__()
        self.model.eval()
        stats_meter = None
        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch
                input_dict = c_loader_iter.next()

                for key, value in input_dict.items():
                    if(not isinstance(value, list)):
                        input_dict[key] = value.to(self.device)

                predictions = self.model(input_dict)
                
                mos_gt = input_dict['sd_labels'][:,0].cpu().numpy()
                mos_est = predictions['mos_est'].max(1)[1].cpu().numpy()
                transformed_points = predictions['transformed_points']

                sel_est = mos_est == 1
                sel_gt = mos_gt == 1
                
                pcd = to_o3d_pcd(transformed_points)
                sd_gt = self.color_dict[mos_gt]
                sd_est = self.color_dict[mos_est]
                pcd_gt = deepcopy(pcd)
                pcd_gt.colors = to_o3d_vec(sd_gt)
                pcd_est = deepcopy(pcd)
                pcd_est.colors = to_o3d_vec(sd_est)
                
                
                VIS_OFFSET = False
                if VIS_OFFSET:
                    offset = predictions['offset_est']
                    offseted_points = transformed_points.clone()
                    offseted_points[:,:2] += offset
                    # 1. visuallise the offsetted points
                    pcd_offset_est = to_o3d_pcd(offseted_points[sel_est])
                    pcd_offset_est.paint_uniform_color(get_blue())
                    
                    pcd_offset_gt = to_o3d_pcd(offseted_points[sel_gt])
                    pcd_offset_gt.paint_uniform_color(get_blue())
                    
                    points_in_range = transformed_points.cpu().numpy()
                    sel = np.logical_and(np.abs(points_in_range[:,0]) < 32, np.abs(points_in_range[:,1]) < 32).astype(np.int)
                    colors = self.color_dict[sel]                
                    pcd_in_range = to_o3d_pcd(points_in_range)
                    pcd_in_range.colors = to_o3d_vec(colors)
                    
                    multi_vis([pcd_est, pcd_gt, pcd_offset_est, pcd_offset_gt],['est','gt','offset_est','offset_gt'])
                    
                
                # 2. visulise the semantic segmentation over the birds eye veiw projection
                VIS_SEM_SEG = True
                if VIS_SEM_SEG:
                    fb_est = predictions['fb_est_per_points'][:,0].cpu().numpy()
                    fb_gt = input_dict['fb_labels'][:,0].cpu().numpy()
        
                    color_fb_est = self.color_dict[fb_est]
                    color_fb_gt = self.color_dict[fb_gt]
                    pcd_fb_est = deepcopy(pcd)
                    pcd_fb_gt = deepcopy(pcd)
                    pcd_fb_est.colors = to_o3d_vec(color_fb_est)
                    pcd_fb_gt.colors = to_o3d_vec(color_fb_gt)
                    
                    multi_vis([pcd_fb_est, pcd_fb_gt],['est','gt'])
                    
                    
                    # occ_map, fb_map = predictions['occ_map'], predictions['fb_seg_gt']
                    # fb_est_map = predictions['fb_seg_est'].max(dim=2, keepdim=True)[1] #[B, T, 1, Ny, Nx]
                    # B,T,_, H, W = occ_map.size()
                    # T = min(T, 5)
                    
                    # # mask un-occupied pillar
                    # fb_est_map = fb_est_map * occ_map
                    

                    # plt.figure(dpi=250, figsize=(10,6))
                    # count = 1 
                    # for i in range(T):
                    #     c_canvas = occ_map[0, i, 0].cpu().numpy()
                    #     plt.subplot(3, T, count)
                    #     plt.imshow(c_canvas)
                    #     count+=1 
                        
                    # for i in range(T):
                    #     c_canvas = fb_map[0, i, 0].cpu().numpy()
                    #     plt.subplot(3, T, count)
                    #     plt.imshow(c_canvas)
                    #     count+=1 
                        
                    # for i in range(T):
                    #     c_canvas = fb_est_map[0, i, 0].detach().cpu().numpy()
                    #     plt.subplot(3, T, count)
                    #     plt.imshow(c_canvas)  
                    #     count+=1 
                    # plt.show()      
                
                VIS_CLUSTER = False
                if VIS_CLUSTER:
                    # 3. visualise the clustering results
                    inst_labels = input_dict['inst_labels'][:,0].cpu().numpy()
                    sd_labels = input_dict['sd_labels'][:,0].cpu().numpy()
                    fb_labels = input_dict['fb_labels'][:,0].cpu().numpy()
                    inst_from_input = predictions['inst_from_input'].cpu().numpy()
                    inst_labels_est = predictions['inst_labels_est'].cpu().numpy()
                    sel = fb_labels == 1
                    pcd = to_o3d_pcd(transformed_points[sel])
                    
                    pcd_inst_from_input = deepcopy(pcd)
                    inst_colors = self.color_dict[inst_from_input[sel] % self.color_dict.shape[0]]
                    pcd_inst_from_input.colors = to_o3d_vec(inst_colors)
                    pcd_inst_labels_est = deepcopy(pcd)
                    inst_colors = self.color_dict[inst_labels_est[sel] % self.color_dict.shape[0]]
                    pcd_inst_labels_est.colors = to_o3d_vec(inst_colors)
                    pcd_inst_gt = deepcopy(pcd)
                    
                    inst_labels[sd_labels==0] = 0
                    inst_colors = self.color_dict[inst_labels[sel] % self.color_dict.shape[0]]
                    pcd_inst_gt.colors = to_o3d_vec(inst_colors)                
                    
                    multi_vis([pcd_est, pcd_inst_labels_est, pcd_inst_from_input, pcd_inst_gt],['mos estimation','inst_labels_est','inst_from_input','inst_from_gt'],render=False)           
                
                # plt.figure(dpi=250, figsize = (5,2))
                # plt.subplot(1,2,1)
                # plt.imshow(mos_gt.cpu().numpy().astype(np.int))
                # plt.title('Ground truth')
                # plt.subplot(1,2,2)
                # plt.imshow(mos_est.cpu().numpy().astype(np.int))
                # plt.title('prediction')
                # plt.show()
        
    
    def save_results_for_vis(self):
        # generate dataloader by scene names
        self.model.eval()
        
        model_name = self.save_dir.split('/')[1]     
        unique_scenes = ['046']  
        for eachscene in tqdm(unique_scenes):
            test_set = DATASETS[self.dataset](self.config, 'test', data_augmentation = False, scene_name = eachscene)
            dloader = torch.utils.data.DataLoader(test_set, 
                                    batch_size=1, 
                                    num_workers= 6, 
                                    collate_fn=collate_fn,
                                    sampler = DatasetSampler(test_set),
                                    pin_memory=False,
                                    drop_last=False)
            num_iter = len(test_set)
            c_loader_iter = dloader.__iter__()
            
            save_dir = f'results/visualisations/{model_name}/{eachscene}'
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
                    inst_label_est = predictions['inst_labels_est']
                    
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
                    
                    with open(f"{save_dir}/{idx}.pkl", "wb") as f:
                        pickle.dump({
                            'input_points': input_dict['input_points'].cpu().numpy(),
                            'time_indice': time_indice.cpu().numpy(),
                            'fb_labels': fb_labels.cpu().numpy(),
                            'sd_labels': sd_labels.cpu().numpy(),
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
                            'est_flow': est_flow.cpu().numpy(),
                            'data_path': input_dict['data_path']
                        }, f)
                    
                    static_epe = round(epe_per_point[sd_labels == 0].mean().item(), 4)
                    dynamic_epe = round(epe_per_point[sd_labels == 1].mean().item(), 4)                         
                    vis = False
                    if vis:
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

                        #multi_vis([pcd_time, pcd_ego, pcd_inst, pcd_rec],['input','ego_motion_compensation','clustering','reconstruction'], render=False)
                        multi_vis([pcd_rec_gt, pcd_rec_est, pcd_inst_est, pcd_inst_gt],['gt rec',f'pcd_ours: static_epe: {static_epe}, dynamic_epe: {dynamic_epe}', 'est inst', 'gt inst'], render=True)

                    
                    
                    
            epe_per_point = np.array(flow_error_list)
            relative_err = np.array(relative_error_list)
            time_indice = np.array(time_indice_list)        
            fb_label = np.array(fb_label_list)
            sd_label = np.array(sd_label_list)
            data = {
                'fb_label': fb_label, 
                'sd_label': sd_label, 
                'epe_per_point': epe_per_point,
                'relative_error': relative_err,
                'time_indice':time_indice
            }
            np.savez_compressed(f'{save_dir}/flow_error', **data)  
        
        
    def save_results_by_scene(self):
        # generate dataloader by scene names
        dset_cfg_path = f'../assets/configs/datasets/{self.dataset}/full_split/test_info.txt'
        dset_cfg = np.loadtxt(dset_cfg_path, dtype = str).tolist()
        scenes = [ele.split('/')[3] for ele in dset_cfg]
        unique_scenes = sorted(list(set(scenes)), key=natural_key)
        self.model.eval()
        
        model_name = self.save_dir.split('/')[1]     
        stats_meter = None         
        for eachscene in tqdm(unique_scenes):
            test_set = DATASETS[self.dataset](self.config, 'test', data_augmentation = False, scene_name = eachscene)
            dloader = torch.utils.data.DataLoader(test_set, 
                                    batch_size=1, 
                                    num_workers= 6, 
                                    collate_fn=collate_fn,
                                    sampler = DatasetSampler(test_set),
                                    pin_memory=False,
                                    drop_last=False)
            num_iter = len(test_set)
            c_loader_iter = dloader.__iter__()
            
            save_dir = f'results/new_ablations/{model_name}/{eachscene}'
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
                    
                    # compute scene flow error
                    input_points = input_dict['input_points']
                    time_indice = input_dict['time_indice'][:,1].long()
                    ego_motion_gt = input_dict['ego_motion_gt'].float()[0]
                    inst_motion_gt = input_dict['inst_motion_gt'][0].to(self.device)
                    inst_label_gt = input_dict['inst_labels'][:,0]
                    fb_labels = input_dict['fb_labels'][:,0]
                    sd_labels = input_dict['sd_labels'][:,0]
                    sd_label_est = predictions['mos_est'].argmax(1)
                    inst_label_est = predictions['inst_labels_est']
                    
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
                    
                    sel = torch.logical_and(time_indice > 0, input_points[:,-1] > Ground_height[self.dataset])
                    time_indice_list.extend(time_indice[sel].cpu().numpy().astype(np.int8))
                    fb_label_list.extend(fb_labels[sel].cpu().numpy().astype(np.bool))
                    sd_label_list.extend(sd_labels[sel].cpu().numpy().astype(np.bool))
                    relative_error_list.extend(relative_err[sel].cpu().numpy().astype(np.float16))
                    flow_error_list.extend(epe_per_point[sel].cpu().numpy().astype(np.float16))
                    
                    # compute mos
                    mos_stats = self.loss.get_mos_loss(predictions, input_dict)
                    if(stats_meter is None):
                        stats_meter = init_stats_meter(mos_stats['metric'])
                    update_stats_meter(stats_meter, mos_stats['metric'])
                    
                    # compute for cluster
                    self.loss.evaluate_cluster(predictions, input_dict)
                    
            epe_per_point = np.array(flow_error_list)
            relative_err = np.array(relative_error_list)
            time_indice = np.array(time_indice_list)        
            fb_label = np.array(fb_label_list)
            sd_label = np.array(sd_label_list)
            data = {
                'fb_label': fb_label, 
                'sd_label': sd_label, 
                'epe_per_point': epe_per_point,
                'relative_error': relative_err,
                'time_indice':time_indice
            }
            np.savez_compressed(f'{save_dir}/flow_error', **data)  
        
        print('Motion segmentation results')
        stats, mos_iou_message = compute_mean_iou_recall_precision(stats_meter, self.mos_mapping)
        self.logger.write(mos_iou_message)
                    
        print('cluster results from offseted points')
        self.loss.cluster_eval_offset.final_eval()

    def test(self):
        self.save_results_by_scene()
        