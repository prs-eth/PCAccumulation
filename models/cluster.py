import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torchsparse.utils.quantize import sparse_quantize
from toolbox.utils import canonicalise_random_indice, to_tensor


def voxel_downsample(points, voxel_size):
    coords = np.round(points / voxel_size)
    _, sel, inverse_map = sparse_quantize(coords,return_index=True,  return_inverse=True)

    return coords, sel, inverse_map

class Cluster(nn.Module):
    def __init__(self, cfg):
        super(Cluster, self).__init__()
        cluster_cfg = cfg['cluster']
        self.min_p_cluster = cluster_cfg['min_p_cluster']    
        self.voxel_size = cluster_cfg['voxel_size']
        self.cluster_estimator = DBSCAN(min_samples=cluster_cfg['min_samples_dbscan'], metric=cluster_cfg['cluster_metric'], eps=cluster_cfg['eps_dbscan'])

    def cluster(self, points, inst_labels = None):
        """
        Input:
            points:         [N,3]
        Output:
            inst_labels:    [N], inst_labels = 0 means background/ignored points
        """
        # 1. cluster the objects
        if inst_labels is None:
            inst_labels = self.cluster_estimator.fit_predict(points)
        else:
            inst_labels = inst_labels.cpu().numpy()

        # 2. Ignore clusters with less than self.min_p_cluster points
        unique_inst_ids = np.unique(inst_labels).tolist()
        for unique_inst_id in unique_inst_ids:
            n_pts_of_this_inst = (inst_labels == unique_inst_id).sum()
            if(n_pts_of_this_inst<self.min_p_cluster):
                inst_labels[inst_labels==unique_inst_id]=-1
        
        assert inst_labels.min()<=0 
        if inst_labels.min()==-1:
            # 3. canonicalise the labels
            inst_labels = np.array(canonicalise_random_indice(inst_labels.tolist()))
        else:
            inst_labels = np.array(canonicalise_random_indice(inst_labels.tolist())) + 1        
        return inst_labels      
    
    
    def cluster_per_batch(self, mos, offset, transformed_points, fb_labels, use_offset):
        """cluster for each batch sample, downsample the point clouds to speed up, we perform clustering in horizontal plane
        Args:
            mos (tensor):                   [N]
            offset (tensor):                [N, 2]
            transformed_points (tensor):    [N, 3]
            fb_labels:                      [N]
        """
        if fb_labels is not None:
            sel = fb_labels == 1
        else:    
            sel = mos == 1
        device = mos.device
        full_inst_labels = torch.zeros(mos.size(0)).long().to(device)
        if sel.sum() > self.min_p_cluster:
            offset_points = transformed_points.clone()
            offset_points[:,:2] += offset
            
            moving_points = transformed_points[sel].cpu().numpy()
            moving_offset_points = offset_points[sel].cpu().numpy()

            if use_offset:
                _, sub_sel, inverse_map = voxel_downsample(moving_offset_points, 0.05)   
                moving_offset_points[:,-1] = 0
                sub_inst_labels = self.cluster(moving_offset_points[sub_sel])[inverse_map]
            else:
                _, sub_sel, inverse_map = voxel_downsample(moving_points, 0.15)
                moving_points[:,-1] = 0
                sub_inst_labels = self.cluster(moving_points[sub_sel])[inverse_map]
                
            full_inst_labels[sel] =to_tensor(sub_inst_labels).long().to(device)
        
        return full_inst_labels
    
    def forward(self,transformed_points,mos,offset,time_indice,results,fb_labels = None, use_offset = True):
        """run clustering over moving points only

        Args:
            transformed_points (tensor):    [N,3]
            mos (tensor):                   [N]
            offset (tensor):                [N, 2]
            time_indcie (tensor):           [N, 2]  [batch_idx, time_idx]
            fb_labels:                      [N]
            inst_from_input:                [N]  0 means background
            inst_from_offset:               [N]
        """
        batch_size = int(time_indice[:,0].max() + 1)
        
        inst_labels_list = []
        for batch_idx in range(batch_size):
            sel = time_indice[:,0] == batch_idx
            if sel.sum():
                if fb_labels is not None:
                    c_inst_labels = self.cluster_per_batch(mos[sel], offset[sel],transformed_points[sel].clone(), fb_labels[sel], use_offset)
                else:
                    c_inst_labels = self.cluster_per_batch(mos[sel], offset[sel],transformed_points[sel].clone(), None, use_offset)
                inst_labels_list.append(c_inst_labels)
                
        inst_labels = torch.cat(inst_labels_list).long()
        results['inst_labels_est'] = inst_labels