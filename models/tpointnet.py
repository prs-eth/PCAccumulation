import torch, os
import torch.nn as nn
import numpy as np
from toolbox.utils import to_tensor, _EPS
from chamfer_distance.chamfer_distance import ChamferDistance
from scipy.spatial.transform import Rotation as R
from toolbox.se3_utils import quat2mat
import torch.nn.functional as F
from toolbox.register_utils import apply_tsfm, reconstruct_sequence
from torch_scatter import scatter

def random_sample(n1,N):
    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)
    return to_tensor(choice)


def batch_quat2mat(pose_est_rep):
    """
    rememeber to normalise the quaternion first
    Input:  
        pose_est_rep:       [N, 7], 
    Output:
        pose_est_tsfm       [N, 4, 4]
    """
    device = pose_est_rep.device
    pose_est_tsfm = []

    quat = pose_est_rep[:,:4]
    quat = F.normalize(quat, p=2, dim = 1)
    trans = pose_est_rep[:,4:]
    
    rot = quat2mat(quat)

    pose_est_tsfm = torch.eye(4)[None].repeat(pose_est_rep.size(0),1,1).to(device)
    pose_est_tsfm[:,:3,:3] = rot
    pose_est_tsfm[:,:3,3] = trans
    return pose_est_tsfm


def batch_mat2quat(pose_gt, centroids):
    """
    Here we recompute ground truth poses for centered point clouds
    Input:  
        pose_gt:        [B, self.n_frame, 4, 4]
        centroids:      [B, 3]
    Output:
        pose_gt_tsfm:   [B x self.n_frame, 4, 4]
        pose_gt_rep:    [B x self.n_frame, 7]
    """
    pose_gt_tsfm = pose_gt.clone()
    n_frames = pose_gt.size(1)
    device = pose_gt.device
    centroids = centroids.repeat_interleave(n_frames, 0).unsqueeze(2)
    pose_gt_tsfm = pose_gt_tsfm.view(-1, 4, 4)
    B = pose_gt_tsfm.size(0)
    diff = torch.matmul((pose_gt_tsfm[:,:3,:3] - torch.eye(3)[None].repeat(B, 1,1).to(device)), centroids).squeeze(2)
    pose_gt_tsfm[:,:3,3] += diff

    pose_gt_rep = []
    pose_gt_tsfm_np = pose_gt_tsfm.cpu().numpy()
    for idx in range(B):
        c_rot, c_trans = pose_gt_tsfm_np[idx,:3,:3],pose_gt_tsfm_np[idx,:3,3]
        r = R.from_matrix(c_rot)
        c_quat = r.as_quat()
        pose_gt_rep.append(np.concatenate((c_quat, c_trans)))
    
    pose_gt_rep = np.array(pose_gt_rep)
    pose_gt_rep = to_tensor(pose_gt_rep).to(device)

    return pose_gt_tsfm, pose_gt_rep


def evaluate_pose(pose_est_rep, pose_gt_rep, weights):
    """
    Input:
        pose_est_rep:   [K, 7]
        pose_gt_rep:    [K, 7]
        count:          [K]
    """
    # 2. compute pose error
    quat_gt, trans_gt = pose_gt_rep[:,:4], pose_gt_rep[:,4:]
    quat_est, trans_est = pose_est_rep[:,:4], pose_est_rep[:,4:]
    quat_est = F.normalize(quat_est, p=2, dim=1)

    diff_quat = quat_gt - quat_est
    diff_trans = trans_gt - trans_est

    rot_loss = (torch.norm(diff_quat,p=2, dim=1) * weights).sum() / (weights.sum() + _EPS)
    trans_loss = (torch.norm(diff_trans, p=2, dim=1) * weights).sum() / (weights.sum() + _EPS)
    
    return rot_loss, trans_loss
    

class BaseModel(nn.Module):
    """
    ICP baseline
    """
    def __init__(self, config):
        super().__init__()
        self.n_frames = config['voxel_generator']['n_sweeps']
        self.chamfer_dist = ChamferDistance()


    def align_frames(self, points, time_indice, poses):
        """
        Input:
            points:             [N,3]
            points_ts_indice:   [N]
            relative_poses:     [n_frames, 4,4]
        """
        points = points.clone()
        for idx in range(self.n_frames):
            sel = time_indice == idx
            if(sel.sum()):
                points[sel] = apply_tsfm(points[sel], poses[idx])
        return points
    
    def get_chamfer_distance(self,est_points, gt_points,weights):
       """
       Compute Chamfer distance
       Input(Torch.tensor):
            est_points:     [N, 3]
            gt_points:      [N, 3]
       """
       dist1, dist2 = self.chamfer_dist(gt_points[None], est_points[None])
       dist1, dist2 = dist1 * weights, dist2 * weights
       loss = (dist1.sum() + dist2.sum()) / 2
       return loss

    def get_l2_distance(self, est_points, gt_points, weights):
       """
       Compute L2 distance
       Input(Torch.tensor):
            est_points:     [N, 3]
            gt_points:      [N, 3]
       """
       l2_distance = torch.norm(est_points-gt_points, dim=1)
       l2_distance = l2_distance * weights
       loss = l2_distance.sum()
       return loss

    def get_alignment_errors(self, points, time_indice, est_poses, gt_poses):
        """
        Compute alignment errors
        Input(Torch.tensor):
                points:         [N, 3]
                time_indice:    [N]
                est_poses:      [n_frames,4,4]
                gt_poses:       [n_frames,4,4]
        """
        est_points = self.align_frames(points, time_indice, est_poses)
        gt_points = self.align_frames(points, time_indice, gt_poses)
        weights = torch.zeros(est_points.size(0)).to(est_points.device)
        weights[time_indice==1] = 1.0
        # weights[time_indice==2] = 1.0
        weights = weights / (weights.sum() + _EPS)

        chamfer_dist = self.get_chamfer_distance(est_points, gt_points, weights)
        l2_dist = self.get_l2_distance(est_points, gt_points, weights)
        return chamfer_dist, l2_dist



class TPointNet(BaseModel):
    '''
    TPointNet
    '''
    def __init__(self, config):
        BaseModel.__init__(self,config)
        self.geo_embed = nn.Sequential(
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 128, bias=True),
        )
        
        self.motion_embed = nn.Sequential(
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
        )
        
        self.pos_embed = nn.Sequential(
            nn.Linear(4, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 128, bias=True),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        self.min_points_per_frame = config['tpointnet']['min_points']



    def forward(self, input_dict):
        """
        TPointnet first centers point cloud using the centroid of the anchor frame
        """
        mos_feat, frame_feats = input_dict['mos_feats'], input_dict['frame_feats']
        points = input_dict['points']
        time_indice, inst_indice = input_dict['time_indice'], input_dict['inst_labels']
        inst_motion_gt = input_dict['inst_motion_gt']
        mos_labels = input_dict['mos_labels']
        K, T, _, _ = inst_motion_gt.size()
        device = mos_feat.device
        frame_indice = (inst_indice * T + time_indice).long()
        
        ##################################################################
        # 1. assign weights based on motion label and number of points in the frame
        count = torch.ones(frame_indice.size(0)).to(device)
        frame_count = scatter(count, frame_indice, dim=0, dim_size = K * T, reduce='sum')   #[K x T]    
        
        frame_weights = (frame_count > self.min_points_per_frame).float() #{0,1}
        
        inst_mos_label = scatter(mos_labels, frame_indice, dim= 0, dim_size = K*T, reduce = 'max') #[K x T]
        mos_weights = torch.ones_like(inst_mos_label)
        mos_weights[inst_mos_label==0] = 0.2   #[0,1]
        
        temporal_weights = (torch.arange(self.n_frames) + 1).to(device).repeat(K) / self.n_frames  #[0,1]
        
        frame_weights = frame_weights * mos_weights * temporal_weights     
        
        
        ##################################################################
        # 2. embed the geometry/motion features
        # embed motion features
        mos_embedding = self.motion_embed(mos_feat)
        mos_embedding = scatter(mos_embedding, inst_indice, dim=0, dim_size = K, reduce = 'max')  #[K, 128]
        
        # embed geometry features
        geo_embedding = self.geo_embed(frame_feats)
        geo_embedding = scatter(geo_embedding, inst_indice, dim = 0, dim_size = K, reduce = 'max') #[K, 128]
        
        # embed each framef
        frame_centroid = scatter(points, frame_indice, dim =0, dim_size = K * T, reduce = 'mean')  #[K*T, 3]
        inst_centroid = frame_centroid[::T] #[K]
        inst_center_per_point = inst_centroid[inst_indice]
        centered_points = points - inst_center_per_point
        
        frame_input = torch.cat((centered_points, time_indice.unsqueeze(-1) / T), dim = 1).float()
        frame_embedding = self.pos_embed(frame_input)

        frame_embedding = scatter(frame_embedding, frame_indice, dim=0, dim_size = K * T, reduce = 'max') #[K x T, 128]
     
     
        ##################################################################   
        # regress the pose for each possible pair
        anchor_embedding = frame_embedding[::T].repeat_interleave(T, 0) #[K x T, 128]
        geo_embedding = geo_embedding.repeat_interleave(T, 0)  #[K x T, 128]
        mos_embedding = mos_embedding.repeat_interleave(T, 0)
        
        regressor_input = torch.cat((geo_embedding, mos_embedding, frame_embedding, anchor_embedding), dim=1)  #[K x T, 512]
        pose_est_rep = self.regressor(regressor_input) #[K x T, 7]
        
        pose_est_tsfm = batch_quat2mat(pose_est_rep)  #[K x T, 4, 4]
        
        ##################################################################  
        # compute loss
        pose_gt_tsfm, pose_gt_rep = batch_mat2quat(inst_motion_gt, inst_centroid)
        
        rec_points_est = reconstruct_sequence(centered_points, time_indice, inst_indice, pose_est_tsfm.view(K, T, 4, 4), T)
        rec_points_gt = reconstruct_sequence(centered_points, time_indice, inst_indice, pose_gt_tsfm.view(K, T, 4, 4), T)
        diff = rec_points_est - rec_points_gt
        
        l1_loss = torch.norm(diff, p=2, dim=1)
        l2_loss = torch.norm(diff, p=1, dim=1)
        frame_l1_loss = scatter(l1_loss, frame_indice, dim=0, dim_size = K * T, reduce = 'mean') # get rid of influence from point density
        frame_l2_loss = scatter(l2_loss, frame_indice, dim=0, dim_size = K * T, reduce = 'mean')
        l1_loss = (frame_l1_loss * frame_weights).sum() / (frame_weights.sum() + _EPS)
        l2_loss = (frame_l2_loss * frame_weights).sum() / (frame_weights.sum() + _EPS)

        rot_loss, trans_loss = evaluate_pose(pose_est_rep, pose_gt_rep, frame_weights)
        ##################################################################  
        # compensate for centering operation
        inst_centroid = inst_centroid.repeat_interleave(T, 0).unsqueeze(-1)
        diff = torch.matmul(torch.eye(3)[None].repeat(K*T,1,1).to(device) - pose_est_tsfm[:,:3,:3], inst_centroid).squeeze(2)
        pose_est_tsfm[:,:3,3] += diff
        pose_est_tsfm = pose_est_tsfm.view(K, T, 4, 4)
        pose_est_tsfm[:,0] = torch.eye(4)[None].repeat(K, 1, 1).to(device)
        
        results = {
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'rot_loss': rot_loss, 
            'trans_loss': trans_loss,
            'inst_est_motion': pose_est_tsfm  #[K, T, 4, 4]
        }
        
        return results