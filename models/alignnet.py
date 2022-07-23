import torch, os
import numpy as np
from toolbox.register_utils import reconstruct_sequence
from models.tpointnet import TPointNet, BaseModel
from toolbox.utils import _EPS, to_o3d_pcd, to_tensor
import open3d as o3d
from torch_scatter import scatter

def update_gt_inst_motion(inst_motion_gt, ego_motion_gt, ego_motion_est):
    """update the ground truth instance motion

    Args:
        inst_motion_gt (tensor): list of [K, T, 4, 4]
        ego_motion_gt (tensor): [B, T, 4, 4]
        ego_motion_est (tensor): [B, T, 4, 4]
    """
    batch_size = len(inst_motion_gt)
    updated_inst_motion_list = []
    device = ego_motion_gt.device
    
    # move data to gpu first
    inst_motion_gt = [ele.to(device).float() for ele in inst_motion_gt]

    for batch_idx in range(batch_size):
        c_inst_motion_gt = inst_motion_gt[batch_idx]  #[K, T, 4, 4]
        c_ego_motion_gt, c_ego_motion_est = ego_motion_gt[batch_idx], ego_motion_est[batch_idx]  #[T, 4, 4]
        
        K = c_inst_motion_gt.size(0)
        c_ego_motion_gt = c_ego_motion_gt[None].repeat(K, 1,1,1).view(-1, 4,4)
        c_ego_motion_est = c_ego_motion_est[None].repeat(K, 1,1,1).view(-1,4,4)
        c_inst_motion_gt = c_inst_motion_gt.view(-1, 4, 4)
        
        c_updated_inst_motion_gt = c_inst_motion_gt @ c_ego_motion_gt @ torch.linalg.inv(c_ego_motion_est)
        c_updated_inst_motion_gt = c_updated_inst_motion_gt.view(K, -1, 4, 4)
        
        updated_inst_motion_list.append(c_updated_inst_motion_gt)
        
    return updated_inst_motion_list

class AlignNet(BaseModel):
    '''
    TPointNet
    '''
    def __init__(self, config):
        BaseModel.__init__(self,config)
        self.alignment = TPointNet(config)
        self.n_iterations = config['tpointnet']['n_iterations']
        self.pc_range = config['voxel_generator']['range']
        self.icp_threshold = config['tpointnet']['icp_threshold']
        self.mode = config['misc']['mode']
        self.refine_with_icp = config['model']['tpointnet_icp']
        

    def run_icp(self, points, time_indice):
        """
        Input(numpy.ndarray):
            points:                     [N, 3]
            time_indice:                [N]
        Return:
            relative_pose_estimates:    [n_frames, 4, 4]
        """

        min_idx = time_indice.min()
        assert min_idx == 0

        # initialise the anchor point cloud
        sel = time_indice == min_idx
        accumulated_pcd = to_o3d_pcd(points[sel])

        relative_pose_estimates = []
        for idx in range(self.n_frames):
            sel = time_indice == idx
            if(sel.sum() and idx!=min_idx):
                # 1. center the point cloud first
                c_points = points[sel]
                c_pcd = to_o3d_pcd(c_points)

                # run ICP
                reg = o3d.pipelines.registration.registration_icp(c_pcd, accumulated_pcd, self.icp_threshold, np.eye(4),
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
                tsfm = reg.transformation
                c_pcd.transform(tsfm)

                # update relative pose estimation
                relative_pose_estimates.append(tsfm)

            else:
                relative_pose_estimates.append(np.eye(4))
        
        relative_pose_estimates = to_tensor(np.array(relative_pose_estimates))
        return relative_pose_estimates
    
    
    def refine_pose_by_icp(self, points, time_indice, inst_labels, pose_est_tsfm):
        """Refine pose with ICP

        Args:
            points (tensor):        [N, 3]
            time_indice (tensor):       [N]
            inst_label (_type_):        [N]
            pose_est_tsfm (_type_):     [K, T, 4, 4]
        """
        n_inst, T, _, _ = pose_est_tsfm.size()
        rec_points = reconstruct_sequence(points, time_indice, inst_labels, pose_est_tsfm, T)        
        assert n_inst == int(inst_labels.max()) + 1
        for inst_idx in range(n_inst):
            sel = inst_labels == inst_idx
            c_points, c_time_indice = rec_points[sel].cpu().numpy(), time_indice[sel].cpu().numpy()
            refined_pose = self.run_icp(c_points, c_time_indice).to(pose_est_tsfm.device).float()
            pose_est_tsfm[inst_idx] = torch.matmul(refined_pose, pose_est_tsfm[inst_idx])
        return pose_est_tsfm

        
    def padding(self,inst_indice, time_indice, inst_motion):
        """
        remove empty instance, adjust instance label, pad points to instance
        Args:
            inst_indice (tensor):   N
            time_indice (tensoro):  N
            inst_motion: (tensor):  [K, T, 4, 4]
            
        Return:
            padding_list:(tensor):  M, indice of the points to be padded
        """
        device = inst_indice.device
        K, T, _, _ = inst_motion.size()
    
        # padding instance
        padding_list = []
        frame_indice = (inst_indice * T + time_indice).long() #[N]
        count = torch.ones(frame_indice.size(0)).to(device)
        frame_count = scatter(count, frame_indice, dim=0, dim_size = K * T, reduce='sum')   #[K x T]
        inst_count = scatter(count, inst_indice, dim=0, dim_size = K, reduce = 'sum') #[K]
        anchor_count = frame_count[::T] #[K]
        
        sel_inst = (anchor_count == 0) & (inst_count >0)
        if sel_inst.sum():
            inst_list = torch.where(sel_inst)[0].tolist()
            for inst_idx in inst_list:
                c_count = frame_count[inst_idx*T:(inst_idx+1)*T]
                assert c_count[0] == 0
                pad_frame_idx = torch.where(c_count>0)[0][0]
                pad_frame_idx = inst_idx * T + pad_frame_idx
                sel_points = frame_indice == pad_frame_idx
                padding_list.append(torch.where(sel_points)[0])
        
        if len(padding_list):
            padding_indice= torch.cat(padding_list)
        else:
            padding_indice = None
        
        # adjust instance label, remove empty_instance
        sel_inst = inst_count > 0
        inst_motion = inst_motion[sel_inst]
        
        mapping = -1 * torch.ones(K).long()
        mapping[sel_inst] = torch.arange(sel_inst.sum())
        mapping = mapping.to(device)
        updated_inst_label = mapping[inst_indice]
        assert updated_inst_label.min() != -1
        
        return padding_indice, inst_motion, updated_inst_label
        
    
    def forward(self, input_dict, results):
        """reconstruct for each instance

        Args:
            input_dict: 
                inst_labels:        [N]
                time_indice:        [N, 2]  [batch_idx, time_indice]
                transformed_points: [N, 3]
                bev_feats:          [B, T, C, Ny, Nx]
                motion_feats:       [B, C, Ny, Nx]
        """
        mos_labels = input_dict['mos_labels']#[N]   
        inst_labels = input_dict['inst_labels'].clone()  #[N]
        time_indice = input_dict['time_indice']  #[N, 2]
        transformed_points = input_dict['transformed_points'].clone() #[N, 3]
        inst_motion_gt = input_dict['inst_motion_gt']  #[[K, T, 4, 4]]
        backbone_feats = input_dict['backbone_feats'] #[N, C]
        mos_feats = input_dict['motion_feats']
        ego_motion_est = input_dict['ego_motion_est']  #[B, T, 4, 4]
        ego_motion_gt = input_dict['ego_motion_gt'] #[B, T, 4, 4]
        
        device = inst_labels.device
        n_points = inst_labels.size(0)
        
        if self.mode == 'test':
            n_instance = inst_labels.max()+1
            inst_motion_gt = [torch.eye(4)[None,None].repeat(n_instance, self.n_frames, 1, 1).to(device)]
        
        # 2. get the ego-motion corrected instance motion
        updated_inst_motion = update_gt_inst_motion(inst_motion_gt, ego_motion_gt, ego_motion_est)
        
        # 3. update instance labels 
        batch_size = len(updated_inst_motion)
        running_idx = 0
        
        for batch_idx in range(batch_size):
            sel = time_indice[:,0] == batch_idx
            if sel.sum():
                inst_labels[sel] += running_idx
                running_idx += updated_inst_motion[batch_idx].size(0)
        updated_inst_motion = torch.cat(updated_inst_motion)  #[K, T, 4, 4]
       
        # 4. remove empty instance, adjust instance label, pad points to instance
        padding_indice, updated_inst_motion, inst_labels = self.padding(inst_labels, time_indice[:,1], updated_inst_motion)
        inst_motion_gt = updated_inst_motion.clone()
        K, T, _, _ = updated_inst_motion.size()
        
        if padding_indice is not None:
            to_pad_time_indice = torch.zeros_like(padding_indice).long()
            padded_time_indice = torch.cat((time_indice[:,1], to_pad_time_indice))
            padding_indice = torch.cat((torch.arange(n_points).to(device).long(), padding_indice))    
        else:
            padding_indice = torch.arange(n_points).to(device).long()
            padded_time_indice = time_indice[:,1]
            
        padded_backbone_feats = backbone_feats[padding_indice]
        padded_mos_feats = mos_feats[padding_indice]
        padded_inst_labels = inst_labels[padding_indice]
        padded_mos_labels = mos_labels[padding_indice]
        padded_transformed_points = transformed_points[padding_indice]
        padded_transformed_points_back = padded_transformed_points.clone()
        # iterative motion regression
        results['tpointnet_loss_terms'] = dict()
        final_pose_est = None
        
        tpointnet_input = {
            'frame_feats': padded_backbone_feats,
            'time_indice': padded_time_indice,
            'inst_labels': padded_inst_labels,
            'mos_labels': padded_mos_labels,
            'mos_feats': padded_mos_feats
        }     
        for idx in range(self.n_iterations):
            tpointnet_input['points'] = padded_transformed_points.detach()
            tpointnet_input['inst_motion_gt'] = updated_inst_motion.detach()
            
            predictions = self.alignment(tpointnet_input)
            results['tpointnet_loss_terms'][f'{idx}_th'] = predictions
            
            # update transformed points
            c_inst_pose_est = predictions['inst_est_motion']  #[K, T, 4, 4]
            padded_transformed_points = reconstruct_sequence(padded_transformed_points, padded_time_indice, padded_inst_labels, c_inst_pose_est, T)
            
            # update ground truth instance pose
            updated_inst_motion = updated_inst_motion.view(-1,4, 4)
            c_inst_pose_est = c_inst_pose_est.view(-1,4,4)
            updated_inst_motion[:,:3,:3] = torch.matmul(updated_inst_motion[:,:3,:3], c_inst_pose_est[:,:3,:3].transpose(1, 2))
            updated_inst_motion[:,:3,3] = updated_inst_motion[:,:3,3] - torch.matmul(updated_inst_motion[:,:3,:3], c_inst_pose_est[:,:3,3].unsqueeze(-1)).squeeze(-1)
            updated_inst_motion = updated_inst_motion.view(K, T, 4, 4)
            
            # accumulate the pose estimation
            if final_pose_est is None:
                final_pose_est = c_inst_pose_est
            else:
                final_pose_est = torch.matmul(c_inst_pose_est, final_pose_est)
                
                
        # refine pose with ICP if needed
        final_pose_est = final_pose_est.view(K, T, 4, 4)
        if self.refine_with_icp:
            final_pose_est = self.refine_pose_by_icp(padded_transformed_points_back, padded_time_indice, padded_inst_labels, final_pose_est)
            
        # compute final error 
        rec_points_est = reconstruct_sequence(input_dict['transformed_points'], time_indice[:,1], inst_labels, final_pose_est, T)
        rec_points_gt = reconstruct_sequence(input_dict['transformed_points'], time_indice[:,1], inst_labels, inst_motion_gt, T)
        
                    
        l2_error = torch.norm(rec_points_est - rec_points_gt, p=2, dim=1)
        weights = time_indice[:,1]> 0
        weights_mos = (input_dict['mos_labels'] == 1) & (time_indice[:,1] > 0)
        l2_error_full = (l2_error * weights).sum() / (weights.sum()  + _EPS)
        l2_error_dynamic = (l2_error * weights_mos).sum() / (weights_mos.sum() + _EPS)
        
        results['inst_l2_error'] = l2_error_full.item()  
        results['dynamic_inst_l2_error'] = l2_error_dynamic.item()
        results['inst_labels_adjusted'] = inst_labels
        results['inst_pose_est'] = final_pose_est
        results['sub_rec_est'] = rec_points_est
                
        return results