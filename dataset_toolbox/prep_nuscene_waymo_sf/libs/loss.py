"""
Loss functions

Author: Shengyu Huang
Last modified: 30.11.2020
"""
import torch
import torch.nn as nn
import numpy as np
from libs.utils import _EPS
from libs.lovasz_softmax import Lovasz_softmax
from libs.outlier_loss import OutlierLoss
from libs.register_utils import ego_motion_compensation, reconstruct_sequence
from torch_scatter import scatter
from libs.cluster_iou_accuracy import ClusterEvaluation
from libs.utils import to_array

Ground_height = {
    'waymo': 0.34,
    'nuscene': -1.54
}

def compute_iou(predictions, gt, n_class, ignore_index):
    """
    We omit the ignore_class
    predictions:    [N]
    gt:             [N]
    n_class:        integer
    """
    intersection_list, union_list, pred_pos_list, gt_pos_list = [],[],[],[]
    for idx in range(n_class):
        if idx!= ignore_index:
            sel_gt = gt == idx
            sel_pred = predictions == idx

            pred_pos_list.append(sel_pred.sum().item() / 1e3)
            gt_pos_list.append(sel_gt.sum().item() / 1e3)

            intersection = (predictions[sel_gt] == idx).sum().item() / 1e3
            intersection_list.append(intersection)
            union = sel_pred.sum().item() / 1e3 + sel_gt.sum().item() / 1e3 - intersection
            union_list.append(union)      
    
    intersection = np.array(intersection_list)
    union = np.array(union_list)
    pred_pos = np.array(pred_pos_list)
    gt_pos = np.array(gt_pos_list)

    stats = {
        'intersection': intersection,
        'union': union,
        'pred_positives': pred_pos,
        'gt_positives': gt_pos
    }

    return stats

class FuseLoss(nn.Module):
    def __init__(self,config):
        super(FuseLoss,self).__init__()
        self.outlier_loss = OutlierLoss()
        self.lovasz_loss = Lovasz_softmax()
        self.n_classes = 2
        self.ignore_index = -1
        self.weights_mode = 'sqrt_inv_freq'
        self.softmax = nn.Softmax(dim = 1)
        self.dataset = config['dataset']
        
        # ego motion
        self.w_pose_l1_loss = config['w_pose_l1_loss']
        self.w_perm_loss = config['w_perm_loss']
        
        # motion segmentation
        self.w_mos_bce_loss = config['w_mos_bce_loss']
        self.w_mos_lovasz_loss = config['w_mos_lovasz_loss']
        
        # foreground/background segmentation
        self.w_fb_bce_loss = config['w_fb_bce_loss']
        self.w_fb_lovasz_loss = config['w_fb_lovasz_loss']
        
        # offset loss
        self.w_offset_norm_loss = config['w_offset_norm_loss']
        self.w_offset_dir_loss = config['w_offset_dir_loss']
        self.supervise_full_foreground_offset = config['offset_full_foreground_objects']
        
        # tpointnet loss
        self.w_obj_loss = config['w_obj_loss']
        self.w_obj_rot_loss = config['w_obj_rot_loss']
        self.w_obj_trans_loss = config['w_obj_trans_loss']
        self.w_obj_l1_loss = config['w_obj_l1_loss']
        self.w_obj_pose_loss = config['w_obj_pose_loss']
        self.obj_gamma = config['obj_gamma']
        
        # evaluate cluster results
        self.cluster_eval_input = ClusterEvaluation(config)
        self.cluster_eval_offset = ClusterEvaluation(config)
        

    def get_ce_weights(self, gt_label, max_weights = 50):
        # get inverse_frequency of each class from ground truth label
        counts =[]
        device = gt_label.device
        for label in range(self.n_classes):
            counts.append((gt_label == label).sum().item()+_EPS)
        counts = torch.tensor(counts).to(device)
        inv_freq = counts.sum() / counts

        if self.weights_mode =='constant':
            seg_weight = torch.ones(inv_freq.size()).to(device)
            seg_weight[1:] = self.pos_weight
        elif self.weights_mode == 'inv_freq':
            seg_weight = torch.clamp(inv_freq, 0, max_weights)
        elif self.weights_mode == 'sqrt_inv_freq':
            seg_weight = torch.clamp(torch.sqrt(inv_freq), 0, max_weights)
        else:
            raise NotImplementedError

        return seg_weight  
    
    def get_seg_loss(self, gt, est):
        """segmentation loss

        Args:
            gt (tensor): [B] long tensor
            est (tensor): [B, C]
        """
        stats = dict()
        
        # compute weights in an online fashion
        seg_weights = self.get_ce_weights(gt)
        criterion = torch.nn.CrossEntropyLoss(weight=seg_weights, ignore_index=self.ignore_index)

        # 1. compute weighted ce loss and lovasz softmax loss
        ce_loss = criterion(est, gt)
        mos_softmax = self.softmax(est)
        lovasz_loss = self.lovasz_loss(mos_softmax, gt)

        stats['bce_loss'] = ce_loss
        stats['lovasz_loss'] = lovasz_loss

        # 2. update intersection, union, recall, precision, 
        predictions = est.argmax(1)
        stats['metric'] = compute_iou(predictions, gt, self.n_classes, self.ignore_index)
        
        return stats


    def get_mos_loss(self,predictions, input_dict):
        """We only supervise the foreground points
        Returns:
            _type_: _description_
        """
        mos_gt, mos_est = input_dict['sd_labels'][:,0].long(), predictions['mos_est']
        fb_gt, fb_est = input_dict['fb_labels'][:,0], predictions['fb_est_per_points'][:,0]
        fb_mask = torch.logical_or(fb_gt == 1, fb_est == 1)
        
        sel_ground = input_dict['input_points'][:,-1] > Ground_height[self.dataset]
        fb_mask = torch.logical_and(fb_mask, sel_ground)
        
        if fb_mask.sum():
            mos_stats = self.get_seg_loss(mos_gt[fb_mask], mos_est[fb_mask])
        else:
            metric = {
                'intersection': np.zeros(2),
                'union': np.zeros(2),
                'pred_positives': np.zeros(2),
                'gt_positives': np.zeros(2)
            }
            mos_stats = {
                'metric': metric,
                'bce_loss': torch.tensor(0., requires_grad=True).to(mos_gt.device),
                'lovasz_loss':torch.tensor(0., requires_grad=True).to(mos_gt.device)
            }
        return mos_stats
    
    def get_fb_loss(self, predictions):
        """FG/BG segmentation loss
        We only supervise the occupied pillar 

        Args:
            predictions (tensor): dict

        Returns:
            _type_: _description_
        """
        fb_seg_est = predictions['fb_seg_est'] #[B, T, 2, Ny, Nx]
        fb_seg_gt = predictions['fb_seg_gt'] #[B, T, 1, Ny, Nx]
        occ_map = predictions['occ_map'] #[B, T, 1, Ny, Nx]
        
        fb_seg_est = fb_seg_est.permute(0,1,3,4,2).contiguous().view(-1,2)
        fb_seg_gt = fb_seg_gt.permute(0,1,3,4,2).contiguous().view(-1)
        occ_map = occ_map.permute(0,1,3,4,2).contiguous().view(-1)
        mask = occ_map == 1  
        
        fb_est = fb_seg_est[mask]
        fb_gt = fb_seg_gt[mask]
        
        fb_stats = self.get_seg_loss(fb_gt, fb_est)
        return fb_stats
   
    
    def get_offset_loss(self, input_dict, predictions):
        # get ground truth reconstruction and instance center
        input_points = input_dict['input_points']
        time_indice = input_dict['time_indice']
        ego_motion_gt = input_dict['ego_motion_gt']
        inst_labels = input_dict['inst_labels'][:,0].long()
        bbox_tsfm = input_dict['inst_motion_gt']
        fb_mask = input_dict['fb_labels'][:,0] == 1
        device = fb_mask.device
        if fb_mask.sum():
            batch_size = len(bbox_tsfm)
            device = input_points.device
            n_frames = ego_motion_gt.size(1)
            rec_points_list = []
            inst_center_list = []
        
            for batch_idx in range(batch_size):
                sel = time_indice[:,0] == batch_idx
                c_inst_label = inst_labels[sel]
                c_points = input_points[sel]
                c_ego_motion_gt = ego_motion_gt[batch_idx]
                c_time_indice = time_indice[sel, 1]
                c_bbox_tsfm = bbox_tsfm[batch_idx].to(device)
                c_ego_compensated_points = ego_motion_compensation(c_points, c_time_indice, c_ego_motion_gt)
                c_rec_points = reconstruct_sequence(c_ego_compensated_points, c_time_indice, c_inst_label, c_bbox_tsfm, n_frames)
                rec_points_list.append(c_rec_points)
                c_inst_center = scatter(c_rec_points, c_inst_label, dim=0, reduce = 'mean') # missing instance will be zero padded           
            
                assert c_inst_center.size(0) == c_inst_label.max()+1 
                inst_center_list.append(c_inst_center[c_inst_label])
                
            rec_points_gt = torch.cat(rec_points_list, dim=0) 
            inst_centers = torch.cat(inst_center_list, dim=0)[:,:2]

            # compute offset loss
            est_ego_compensated_pts = predictions['transformed_points'][:,:2] #[N, 2]
            gt_offset = (inst_centers - est_ego_compensated_pts)  #[N, 2]
            gt_offset = gt_offset[fb_mask]
            est_offset = predictions['offset_est'][fb_mask]
            
            # l1 loss over the offset prediction
            offset_norm_loss = torch.abs(gt_offset - est_offset).mean(dim=0).sum()
            offset_l2_error = torch.norm(gt_offset - est_offset, p=2, dim=1).mean().item()
            
            # direction loss over normalised offset prediction
            gt_offset_norm = torch.norm(gt_offset, dim=1, p=2)
            norm_gt_offset = gt_offset / (gt_offset_norm.unsqueeze(-1) + _EPS)
            est_offset_norm = torch.norm(est_offset, dim=1, p=2)
            norm_est_offset = est_offset / (est_offset_norm.unsqueeze(-1) + _EPS)
            direction_diff = 1 - (norm_gt_offset * norm_est_offset).sum(-1)   # (N)
            offset_dir_loss = direction_diff.mean()
            predictions['offset_gt'] = gt_offset
        else:
            offset_norm_loss, offset_dir_loss, offset_l2_error = torch.tensor(0., requires_grad=True).to(device),torch.tensor(0., requires_grad=True).to(device), 0
        
        return offset_norm_loss, offset_dir_loss, offset_l2_error
    
    
    def get_tpointnet_loss(self, predictions):
        total_loss = 0
        n_th = 1
        n_iterations = len(predictions['tpointnet_loss_terms'])
        for key, value in predictions['tpointnet_loss_terms'].items():
            pose_loss = self.w_obj_trans_loss * value['trans_loss'] + self.w_obj_rot_loss * value['rot_loss']
            c_loss = self.w_obj_l1_loss * value['l1_loss'] + self.w_obj_pose_loss * pose_loss
            total_loss += c_loss * self.obj_gamma**(n_iterations - n_th)
            n_th +=1
        
        return total_loss
    
    
    def evaluate_cluster(self, predictions, input_dict):
        time_indice = input_dict['time_indice']
        inst_gt = input_dict['inst_labels'][:,0]
        inst_est_from_offset =  predictions['inst_labels_est']
        mos_label = input_dict['sd_labels'][:,0].float()
        
        sel_ground = input_dict['input_points'][:,-1] > Ground_height[self.dataset]
        batch_size = int(time_indice[:,0].max() + 1)
        for batch_idx in range(batch_size):
            sel = time_indice[:,0] == batch_idx
            sel = torch.logical_and(sel, sel_ground)
            
            # self.cluster_eval_input(inst_est_from_input[sel], inst_gt[sel], mos_label[sel])
            self.cluster_eval_offset(inst_est_from_offset[sel], inst_gt[sel], mos_label[sel])  
            

    
    def forward(self, predictions, input_dict):
        stats = dict()
        total_loss = 0
        
        # ego motion loss
        ego_motion_l1_loss = self.w_pose_l1_loss * predictions['ego_l1_loss']
        total_loss += ego_motion_l1_loss
        stats['ego_l1_loss'] = ego_motion_l1_loss
        stats['ego_l2_loss'] = predictions['ego_l2_loss']
        stats['ego_rot_error'] = predictions['ego_rot_error']
        stats['ego_trans_error'] = predictions['ego_trans_error']
        
        perm_loss = self.outlier_loss(predictions['perm_matrix']) * self.w_perm_loss
        total_loss += perm_loss       
        stats['perm_loss'] = perm_loss  
        
        # foreground/background segmentation loss
        if 'fb_seg_est' in predictions.keys():
            fb_stats = self.get_fb_loss(predictions)        
            fb_loss = self.w_fb_bce_loss * fb_stats['bce_loss'] + self.w_fb_lovasz_loss * fb_stats['lovasz_loss']
            total_loss += fb_loss      
            stats['fb_loss'] = fb_loss  
            stats['fb_metric'] = fb_stats['metric']
        
        # motion segmentation loss
        mos_stats = self.get_mos_loss(predictions, input_dict)
        mos_loss = self.w_mos_bce_loss * mos_stats['bce_loss'] + self.w_mos_lovasz_loss * mos_stats['lovasz_loss']
        total_loss += mos_loss
        stats['mos_loss'] = mos_loss
        stats['mos_metric'] = mos_stats['metric']
        
        # offset prediction loss
        offset_norm_loss, offset_dir_loss, offset_l2_error = self.get_offset_loss(input_dict, predictions)
        offset_loss = offset_dir_loss * self.w_offset_dir_loss + offset_norm_loss * self.w_offset_norm_loss
        total_loss += offset_loss
        stats['offset_loss'] = offset_loss
        stats['offset_l1_loss'] = offset_norm_loss
        stats['offset_dir_loss'] = offset_dir_loss
        stats['offset_l2_error'] = offset_l2_error
        
        if 'tpointnet_loss_terms' in predictions.keys():
            obj_loss = self.get_tpointnet_loss(predictions) * self.w_obj_loss
            total_loss += obj_loss
            stats['obj_loss'] = obj_loss
            stats['inst_l2_error'] = predictions['inst_l2_error']
            stats['dynamic_inst_l2_error'] = predictions['dynamic_inst_l2_error']
            
        # self.evaluate_cluster(predictions, input_dict)
        
        stats['loss'] = total_loss
        return stats