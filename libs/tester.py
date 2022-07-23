import torch
import numpy as np
from toolbox.utils import natural_key, makedirs, _EPS
from toolbox.metrics import init_stats_meter, update_stats_meter, compute_mean_iou_recall_precision

from tqdm import tqdm
from libs.trainer import BaseTrainer
from toolbox.register_utils import ego_motion_compensation, reconstruct_sequence
from libs.dataset import DatasetSampler, NuSceneDataset, WaymoDataset
from libs.dataloader import collate_fn
DATASETS = {
    'nuscene': NuSceneDataset,
    'waymo': WaymoDataset
}


class SegTrainer(BaseTrainer):
    def __init__(self, args):
        BaseTrainer.__init__(self,args)
        self.n_frames = args['data']['n_frames']
        self.dataset = args['data']['dataset']    
        
    def test(self):
        # generate dataloader by scene names
        dset_cfg_path = f'assets/configs/datasets/{self.dataset}/test_info.txt'
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
            
            save_dir = f'results/{model_name}/{eachscene}'
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
                    
                    ego_comp_gt = ego_motion_compensation(input_points, time_indice, ego_motion_gt)
                    full_rec_gt = reconstruct_sequence(ego_comp_gt, time_indice, inst_label_gt, inst_motion_gt, self.n_frames)
                    
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