import torch
import torch.nn as nn
from toolbox.register_utils import kabsch_transformation_estimation, get_relative_pose_torch, rotation_error, translation_error
from toolbox.utils import _EPS
from toolbox.utils import to_o3d_pcd, to_tensor, square_distance
import open3d as o3d
import numpy as np

def refine_pose_with_icp(src_pcd, tgt_pcd, initial_pose, icp_threshold, max_iterations):
    """
    Input: (torch.Tensor) 
        coor_src:       [N, 3]
        coor_tgt:       [M, 3]
        initial_pose:   [4,4]
    Return: 
        refined_pose    [4,4]
    """
    device = initial_pose.device
    initial_pose = initial_pose.detach().cpu().numpy()
    src_pcd.transform(initial_pose)
    reg = o3d.pipelines.registration.registration_icp(src_pcd, tgt_pcd, icp_threshold, np.eye(4),
                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    tsfm = np.array(reg.transformation)
    refined_pose = tsfm @ initial_pose
    refined_pose = to_tensor(refined_pose).to(device).float()

    return refined_pose

class EgoMotionHead(nn.Module):
    """
    Class defining EgoMotionHead
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        self.slack = config['pose_estimation']['add_slack']
        self.sinkhorn_iter = config['pose_estimation']['sinkhorn_iter']

        # Affinity parameters
        self.beta = torch.nn.Parameter(torch.tensor(-5.0))
        self.alpha = torch.nn.Parameter(torch.tensor(-5.0))

        self.softplus = torch.nn.Softplus()
        self.ego_n_points =config['pose_estimation']['n_kpts']
        
        self.frequence = config['data']['freq']
        self.n_sweeps = config['voxel_generator']['n_sweeps']
        self.ego_max_speed = config['data']['max_speed']
        self.dataset = config['data']['dataset']
        self.icp_threshold = config['pose_estimation']['icp_threshold']
        self.icp_max_iter = config['pose_estimation']['icp_max_iter']
        self.refine_with_icp = config['model']['ego_icp']

        self.seq_pose = config['pose_estimation']['seq_pose']
        
        if self.seq_pose == 'chain':
            self.seq_pose_est = self.sequence_pose_est_chain
        elif self.seq_pose == 'skip':
            self.seq_pose_est = self.sequence_pose_est_skip
        else:
             self.seq_pose_est = self.sequence_pose_est_full
        


    def compute_rigid_transform(self, xyz_s, xyz_t, weights):
        """Compute rigid transforms between two point sets
        Args:
            a (torch.Tensor): (B, M, 3) points
            b (torch.Tensor): (B, N, 3) points
            weights (torch.Tensor): (B, M)
        Returns:
            Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
        """

        weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
        centroid_s = torch.sum(xyz_s * weights_normalized, dim=1)
        centroid_t = torch.sum(xyz_t * weights_normalized, dim=1)
        s_centered = xyz_s - centroid_s[:, None, :]
        t_centered = xyz_t - centroid_t[:, None, :]
        cov = s_centered.transpose(-2, -1) @ (t_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_s[:, :, None] + centroid_t[:, :, None]

        transform = torch.cat((rot_mat, translation), dim=2)

        return transform

    def sinkhorn(self, log_alpha, n_iters=5, slack=True):
        """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)
        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        """

        # Sinkhorn iterations

        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)


        log_alpha = log_alpha_padded[:, :-1, :-1]

        return log_alpha
    
    
    def pairwise_ego_motion_estimation(self, feats_s, feats_t, coor_s, coor_t, duration):
        """
        Given
        Input:  
            feats_s:    [N,C]
            feats_t:    [M,C]
            coor_s:     [N,3]
            coor_t:     [M,3]
            duration:   float, duration between two frames
        Return:
            perm_matrix:    [1, m, m]
            pose_est:       [4,4]
        """

        # sample fixed number of interest points
        n_source, n_target = feats_s.size(0), feats_t.size(0)
        if n_source > self.ego_n_points:
            choice_source = torch.randperm(n_source)[:self.ego_n_points]
        else:            
            choice_source = torch.arange(self.ego_n_points)
            choice_source[n_source:] = n_source-1
            
        if n_target > self.ego_n_points:
            choice_target = torch.randperm(n_target)[:self.ego_n_points]
        else:
            choice_target = torch.arange(self.ego_n_points)
            choice_target[n_target:] = n_target - 1
        

        feats_s_ego, coor_s_ego = feats_s[choice_source][None], coor_s[choice_source][None]
        feats_t_ego, coor_t_ego = feats_t[choice_target][None], coor_t[choice_target][None]

        # Force transport to be zero for points further than 4 m apart
        threshold_distance = duration * self.ego_max_speed
        support_ego = (square_distance(coor_s_ego, coor_t_ego, normalised=False) < threshold_distance**2).float()
        
        # Cost matrix in the feature space
        feat_dist = square_distance(feats_s_ego, feats_t_ego, normalised=True)

        # run OT+SVD for pose estimation     
        affinity = -(feat_dist - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)
        log_perm_matrix = self.sinkhorn(affinity, n_iters=self.sinkhorn_iter, slack=self.slack)

        perm_matrix = torch.exp(log_perm_matrix) * support_ego
        weighted_t = perm_matrix @ coor_t_ego / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)
        R_est, t_est, _, _ = kabsch_transformation_estimation(coor_s_ego, weighted_t, weights=torch.sum(perm_matrix, dim=2))      
        R_est, t_est = R_est[0], t_est[0]

        pose_est = torch.eye(4).to(feats_s.device).type(t_est.dtype)
        pose_est[:3,:3] = R_est
        pose_est[:3,3] = t_est[:,0]

        return pose_est, perm_matrix
    
    
    def sequence_pose_est_full(self,  points_list, feats_list, bg_mask_list, c_ego_motion_gt, T, perm_matrix_list, relative_pose_est_list, relative_pose_gt_list, chained_pose_est_list, chained_pose_gt_list):
        """
        We estimate the full pairwise transformation matrice
        """
        anchor_mask = bg_mask_list[0]
        anchor_points = points_list[0]
        anchor_points_est = anchor_points[anchor_mask]
        anchor_feats = feats_list[0][anchor_mask]
        
        identity_pose = torch.eye(4).to(feats_list[0].device)
        relative_pose_est_list.append(identity_pose)
        relative_pose_gt_list.append(identity_pose)
        chained_pose_est_list.append(identity_pose)
        chained_pose_gt_list.append(identity_pose)
        
        total_l1_loss, total_l2_loss = 0,0
        pair_list = []
        for gap in range(1, T):
            for frame_idx in range(T-1):
                anchor_idx = frame_idx
                ref_idx = anchor_idx + gap
                if ref_idx < T:
                    pair_list.append([anchor_idx, ref_idx])
        count = 0
        for eachpair in pair_list:
            anchor_idx, ref_idx = eachpair[0], eachpair[1]
            duration = abs(anchor_idx - ref_idx) / self.frequence
            
            anchor_points, anchor_feats = points_list[anchor_idx], feats_list[anchor_idx]
            ref_points, ref_feats = points_list[ref_idx], feats_list[ref_idx]
            anchor_mask, ref_mask = bg_mask_list[anchor_idx], bg_mask_list[ref_idx]
            
            anchor_points_est, anchor_feats = anchor_points[anchor_mask], anchor_feats[anchor_mask]
            ref_points_est, ref_feats = ref_points[ref_mask], ref_feats[ref_mask]
            
            pose_est, perm_matrix = self.pairwise_ego_motion_estimation(ref_feats, anchor_feats, ref_points_est, anchor_points_est, duration)
            pose_gt = get_relative_pose_torch(c_ego_motion_gt[ref_idx], c_ego_motion_gt[anchor_idx], self.dataset)
            points = torch.cat([ref_points, torch.ones((ref_points.size(0),1)).to(ref_points.device)], dim=1)
            pc_est = (pose_est @ points.T).T[:,:3]
            pc_gt = (pose_gt @ points.T).T[:,:3]
            l1_loss = torch.norm(pc_est - pc_gt, p=1, dim=1).mean()
            l2_loss = torch.norm(pc_est - pc_gt, p=2, dim=1).mean()
            total_l1_loss += l1_loss
            total_l2_loss += l2_loss
            count += 1
            
            if anchor_idx == 0:
                chained_pose_est_list.append(pose_est)
                chained_pose_gt_list.append(pose_gt)
                relative_pose_gt = get_relative_pose_torch(c_ego_motion_gt[ref_idx], c_ego_motion_gt[ref_idx-1], self.dataset)
                relative_pose_gt_list.append(relative_pose_gt)
        
                relative_pose_est = get_relative_pose_torch(chained_pose_est_list[-1], chained_pose_est_list[-2], self.dataset)
                relative_pose_est_list.append(relative_pose_est)
                perm_matrix_list.append(perm_matrix)
        
        return total_l1_loss, total_l2_loss, count
    
    
    def sequence_pose_est_chain(self,  points_list, feats_list, bg_mask_list, c_ego_motion_gt, T, perm_matrix_list, relative_pose_est_list, relative_pose_gt_list, chained_pose_est_list, chained_pose_gt_list):
        """
        Pairwise registration for a sequence by chaining the poses
        """

        anchor_mask = bg_mask_list[0]
        anchor_points = points_list[0][anchor_mask] 
        
        identity_pose = torch.eye(4).to(feats_list[0].device)
        relative_pose_est_list.append(identity_pose)
        relative_pose_gt_list.append(identity_pose)
        chained_pose_est_list.append(identity_pose)
        chained_pose_gt_list.append(identity_pose)
        
        chained_pose = identity_pose
        total_l1_loss, total_l2_loss = 0,0
        for frame_idx in range(T-1):
            src_idx, tgt_idx = frame_idx + 1, frame_idx
                            
            src_feats, tgt_feats = feats_list[src_idx], feats_list[tgt_idx]
            src_points, tgt_points = points_list[src_idx], points_list[tgt_idx]
            duration = 1.0 / self.frequence
            
            # sample the background 
            src_bkgd_mask, tgt_bkgd_mask = bg_mask_list[src_idx], bg_mask_list[tgt_idx]
            
            src_feats, tgt_feats = src_feats[src_bkgd_mask], tgt_feats[tgt_bkgd_mask]
            src_points_est, tgt_points_est = src_points[src_bkgd_mask], tgt_points[tgt_bkgd_mask]
            
            # run pairwise registration
            pose_est, perm_matrix = self.pairwise_ego_motion_estimation(src_feats, tgt_feats, src_points_est, tgt_points_est, duration)

            perm_matrix_list.append(perm_matrix)
            relative_pose_est_list.append(pose_est)
            
            # update chained pose
            chained_pose = chained_pose @ pose_est
            chained_pose_est_list.append(chained_pose)
            
            # compute l1/l2 loss
            pose_gt = get_relative_pose_torch(c_ego_motion_gt[src_idx], c_ego_motion_gt[tgt_idx], self.dataset)
            relative_pose_gt_list.append(pose_gt)
            chained_pose_gt_list.append(get_relative_pose_torch(c_ego_motion_gt[src_idx], c_ego_motion_gt[0], self.dataset))
            
            points = torch.cat([src_points, torch.ones((src_points.size(0),1)).to(src_points.device)], dim=1)
            pc_est = (pose_est @ points.T).T[:,:3]
            pc_gt = (pose_gt @ points.T).T[:,:3]
            l1_loss = torch.norm(pc_est - pc_gt, p=1, dim=1).mean()
            l2_loss = torch.norm(pc_est - pc_gt, p=2, dim=1).mean()
            total_l1_loss += l1_loss
            total_l2_loss += l2_loss
        
        return total_l1_loss, total_l2_loss, T-1
        

    def sequence_pose_est_skip(self,  points_list, feats_list, bg_mask_list, c_ego_motion_gt, T, perm_matrix_list, relative_pose_est_list, relative_pose_gt_list, chained_pose_est_list, chained_pose_gt_list):
        """
        Pairwise registration for a sequence by estimating the pose for each reference frame to the anchor frame
        """
        anchor_mask = bg_mask_list[0]
        anchor_points = points_list[0]
        anchor_points_est = anchor_points[anchor_mask]
        anchor_feats = feats_list[0][anchor_mask]
        
        identity_pose = torch.eye(4).to(feats_list[0].device)
        relative_pose_est_list.append(identity_pose)
        relative_pose_gt_list.append(identity_pose)
        chained_pose_est_list.append(identity_pose)
        chained_pose_gt_list.append(identity_pose)
        
        total_l1_loss, total_l2_loss = 0,0
        for frame_idx in range(T-1):
            ref_idx = frame_idx + 1
            ref_points, ref_feats = points_list[ref_idx], feats_list[ref_idx]
            ref_bkgd_mask = bg_mask_list[ref_idx]
            duration = (frame_idx+1) / self.frequence
            
            # pairwise registration
            ref_feats, ref_points_est = ref_feats[ref_bkgd_mask], ref_points[ref_bkgd_mask]
            pose_est, perm_matrix = self.pairwise_ego_motion_estimation(ref_feats, anchor_feats, ref_points_est, anchor_points_est, duration)
            pose_gt = get_relative_pose_torch(c_ego_motion_gt[ref_idx], c_ego_motion_gt[0], self.dataset)
            
            # update the chained pose
            perm_matrix_list.append(perm_matrix)
            chained_pose_est_list.append(pose_est)
            chained_pose_gt_list.append(pose_gt)
        
            # compute l1/l2 loss            
            points = torch.cat([ref_points, torch.ones((ref_points.size(0),1)).to(ref_points.device)], dim=1)
            pc_est = (pose_est @ points.T).T[:,:3]
            pc_gt = (pose_gt @ points.T).T[:,:3]
            l1_loss = torch.norm(pc_est - pc_gt, p=1, dim=1).mean()
            l2_loss = torch.norm(pc_est - pc_gt, p=2, dim=1).mean()
            total_l1_loss += l1_loss
            total_l2_loss += l2_loss
            
            # update relative pose
            relative_pose_gt = get_relative_pose_torch(c_ego_motion_gt[ref_idx], c_ego_motion_gt[ref_idx-1], self.dataset)
            relative_pose_gt_list.append(relative_pose_gt)
    
            relative_pose_est = get_relative_pose_torch(chained_pose_est_list[-1], chained_pose_est_list[-2], self.dataset)
            relative_pose_est_list.append(relative_pose_est)
        
        return total_l1_loss, total_l2_loss, T-1
    
    
    def pose_refinement(self, pillar_points, bg_mask_list, init_pose):
        """
        pillar_points: [[M, 3]] list of points
        init_pose:      
        """    
        T = len(pillar_points)
        anchor_points = pillar_points[0]
        anchor_mask = bg_mask_list[0]
        anchor_points_est = anchor_points[anchor_mask]
        anchor_pcd = to_o3d_pcd(anchor_points_est)
        
        identity_pose = torch.eye(4).to(anchor_points.device)
        refined_pose_list = []
        refined_pose_list.append(identity_pose)
        for frame_idx in range(1, T):
            ref_points = pillar_points[frame_idx]
            ref_mask = bg_mask_list[frame_idx]
            ref_points_est = ref_points[ref_mask]
            c_init_pose = init_pose[frame_idx]
            ref_pcd = to_o3d_pcd(ref_points_est)
            refined_pose = refine_pose_with_icp(ref_pcd, anchor_pcd, c_init_pose, self.icp_threshold, self.icp_max_iter)
            
            refined_pose_list.append(refined_pose)

        return refined_pose_list


    def forward(self, bev_feats, fb_est, occ_map, pts_mean_map, ego_motion_gt, input_points, fb_est_per_point, time_indice, results):
        """Pairwise registration
        Args:
            bev_feats (tensor):         [B, T, C, Ny, Nx]
            fb_est (tensor):            [B, T, 1, Ny, Nx]  # 1 means foreground, we should use background for registration
            occ_map (tensor):           [B, T, 1, Ny, Nx]  # 1 means this pillar is occupied
            pts_mean_map (tensor):      [B, T, 3, Ny, Nx]
            ego_motion_gt:              [B, n_frame, 4, 4]
        """
        B, T, C, Ny, Nx= bev_feats.size()

        total_l1_loss = 0
        total_l2_loss = 0
        count = 0
        
        perm_matrix_list = []
        relative_pose_est_list= [] # between consecutive frames
        relative_pose_gt_list = []
        chained_pose_est_list = []
        chained_pose_gt_list = []
        
        # loop over the batch 
        for batch_idx in range(B):
            c_ego_motion_gt = ego_motion_gt[batch_idx]
            bg_mask_list = []
            feats_list = []  # sampled per-point faeture
            points_list = [] # sampled point coordinates, here we use the mean coordinate within each voxel without loss of granuity 
            # Sample non-empty voxels and associated features 
            raw_points_list = []
            raw_bg_mask_list = []
            
            for frame_idx in range(T):
                is_occupied = occ_map[batch_idx, frame_idx,0] #[Ny, Nx]
                is_occupied = is_occupied.view(-1) > 0  # mask of the occupied pillar
                c_feats_map = bev_feats[batch_idx, frame_idx] #[C, Ny, Nx]
                c_pts_map = pts_mean_map[batch_idx, frame_idx] #[3, Ny, Nx]
                this_points = c_pts_map.permute(1,2,0).view(Ny * Nx, 3)[is_occupied]
                this_feats = c_feats_map.permute(1,2,0).view(Ny * Nx, C)[is_occupied]
                
                points_list.append(this_points)
                feats_list.append(this_feats)
                
                c_est_background_mask = fb_est[batch_idx, frame_idx, 0].view(-1) == 0
                bg_mask_list.append(c_est_background_mask[is_occupied])
                
                sel = (time_indice[:,0] == batch_idx) & (time_indice[:,1] == frame_idx)
                raw_points_list.append(input_points[sel])
                raw_bg_mask_list.append(fb_est_per_point[sel, 0] == 0)
            
            # estimate the pose for this sequence
            c_l1_loss, c_l2_loss, c_count = self.seq_pose_est(points_list, feats_list, bg_mask_list, c_ego_motion_gt, T, perm_matrix_list, relative_pose_est_list, relative_pose_gt_list, chained_pose_est_list, chained_pose_gt_list)
            
            if self.refine_with_icp:
                refined_pose = self.pose_refinement(raw_points_list, raw_bg_mask_list, chained_pose_est_list[-T:])
                chained_pose_est_list[-T:] = refined_pose
            
            total_l1_loss += c_l1_loss
            total_l2_loss += c_l2_loss
            count += c_count
            
                
        avg_l1_loss = total_l1_loss / count     
        avg_l2_loss = total_l2_loss / count 
        chained_pose_est = torch.stack(chained_pose_est_list)
        chained_pose_gt = torch.stack(chained_pose_gt_list)
        
        
        rot_est, rot_gt = chained_pose_est[:,:3,:3], chained_pose_gt[:,:3,:3]
        trans_est, trans_gt = chained_pose_est[:,:3,3].unsqueeze(-1), chained_pose_gt[:,:3,3].unsqueeze(-1)
        rot_error, trans_error = rotation_error(rot_est,rot_gt).mean().item(), translation_error(trans_est, trans_gt).mean().item()
        rot_error = rot_error * self.n_sweeps / (self.n_sweeps - 1)
        trans_error = trans_error * self.n_sweeps / (self.n_sweeps - 1)
        
        chained_pose_est = chained_pose_est.view(B, T, 4, 4)
        chained_pose_gt = chained_pose_gt.view(B, T, 4, 4)
        
        results['ego_l1_loss'] = avg_l1_loss
        results['ego_l2_loss'] = avg_l2_loss
        results['ego_rot_error'] = rot_error
        results['ego_trans_error'] = trans_error
        results['perm_matrix'] = perm_matrix_list
        results['ego_motion_est'] = chained_pose_est
        results['ego_motion_gt'] = chained_pose_gt