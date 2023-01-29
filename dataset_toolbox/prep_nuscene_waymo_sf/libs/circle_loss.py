import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.utils import square_distance
import numpy as np
from libs.register_utils import apply_tsfm

class CircleLoss(nn.Module):
    def __init__(self, args):
        super(CircleLoss,self).__init__()

        self.log_scale = args['circle_loss']['log_scale']
        self.pos_optimal = args['circle_loss']['pos_optimal']
        self.neg_optimal = args['circle_loss']['neg_optimal']

        self.pos_margin = args['circle_loss']['pos_margin']
        self.neg_margin = args['circle_loss']['neg_margin']
        self.max_points = args['circle_loss']['max_points']

        self.pos_radius = args['data']['voxel_size'] * 2.5
        self.safe_radius = args['data']['voxel_size'] * 4

        self.sample_points = 3036


    def get_recall(self,coords_dist,feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall


    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Input:
            coords_dist:        [N, N]
            feats_dist:         [N, N]
        """
        pos_mask = coords_dist < self.pos_radius 
        neg_mask = coords_dist > self.safe_radius 

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    
    def forward(self,feats_s, feats_t, coor_s, coor_t, pose_gt):
        """
        Input:
            feats_s:    [N, C]
            feats_t:    [M, C]
            coor_s:     [N, 3]
            coor_t:     [M, 3]
            pose_gt:    [4, 4]
        """
        # we first transform the point clouds
        coor_s = apply_tsfm(coor_s, pose_gt)

        # we first randomly sample 2048 points
        n_source, n_target = feats_s.size(0), feats_t.size(0)
        choice_source = torch.randperm(n_source)[:self.sample_points]
        choice_target = torch.randperm(n_target)[:self.sample_points]

        feats_s_ego, coor_s_ego = feats_s[choice_source], coor_s[choice_source]
        feats_t_ego, coor_t_ego = feats_t[choice_target], coor_t[choice_target]

        # then select correpondence from them
        coor_dist = square_distance(coor_s_ego[None], coor_t_ego[None], normalised=False).squeeze(0)
        sel = torch.where(coor_dist < (self.pos_radius - 0.001)**2)
        src_idx, tgt_idx = sel[0], sel[1]
        randperm = torch.randperm(src_idx.size(0))
        src_idx, tgt_idx = src_idx[randperm][:self.max_points], tgt_idx[randperm][:self.max_points]

        feats_s_ego, coor_s_ego = feats_s_ego[src_idx], coor_s_ego[src_idx]
        feats_t_ego, coor_t_ego = feats_t_ego[tgt_idx], coor_t_ego[tgt_idx]

        # compute circle loss
        coords_dist = torch.sqrt(square_distance(coor_s_ego[None], coor_t_ego[None], normalised=False).squeeze(0))
        feats_dist = torch.sqrt(square_distance(feats_s_ego[None], feats_t_ego[None], normalised=True).squeeze(0))

        recall = self.get_recall(coords_dist, feats_dist)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        return recall, circle_loss