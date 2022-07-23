import torch.nn.functional as F
import torch.nn as nn
from models.pillar_encoder import PillarFeatureNet, scatter_point_pillar, inverse_scatter_point_pillar, temporal_ungrid, ungrid
from models.unet import SegHead2D, UNet
from models.stpn import STPN
import torch, os
from models.egomotion import EgoMotionHead
from torch_scatter import scatter
from models.alignnet import AlignNet
from models.cluster import Cluster
MIN_POINTS = 15

class MotionNet(nn.Module):
    def __init__(self, cfg):
        super(MotionNet, self).__init__()    
        pillar_encoder_cfg = cfg['pillar_encoder'] 
        unet_cfg = cfg['unet']
        
        # backbone 
        self.pillar_encoder = PillarFeatureNet(pillar_encoder_cfg)
        self.unet = UNet(**unet_cfg)
        
        # segmentation head
        self.semseg_head = SegHead2D(unet_cfg['in_channels'], 2)
        
        # ego motion
        self.ego_feats_head = SegHead2D(unet_cfg['in_channels'], cfg['pose_estimation']['feats_dim'])
        self.ego_motion_head = EgoMotionHead(cfg) 

        self.resolution = cfg['voxel_generator']['voxel_size']
        self.pc_range = cfg['voxel_generator']['range']
        
        # motion segmentation and offset prediction
        self.motionhead = STPN(cfg['stpn']['feat_dim'])
        
        # clustering module
        self.cluster = Cluster(cfg)
        
        self.mode = cfg['misc']['mode']
        
        # per-instance motion modelling
        self.reconstructor = AlignNet(cfg)

    
    def get_transformed_grid(self, pose, H, W, x_reso, y_reso, x_min, y_min):
        """Sample a grid for grid_sampling

        Args:
            pose:       [4, 4]
            H:          Height of feature map
            W:          Width of feature map
            x_reso:     resolution along x axis
            y_reso:     resolution along y axis
            x_min:      min x coordinate
            y_min:      min y coordinate
        Returns:
            grid:       [2, H, W] in the range of [-1,1]
        """
        # mesh grid 
        device = pose.device
        xx = torch.arange(0, W).view(1,-1).repeat(H,1) + 0.5
        yy = torch.arange(0, H).view(-1,1).repeat(1,W) + 0.5
        xx = xx.view(1,H,W)
        yy = yy.view(1,H,W)
        grid = torch.cat((xx,yy),0).float()   #[2, H, W] 
        
        # convert to world coordinate
        grid[0] = grid[0] * x_reso + x_min
        grid[1] = grid[1] * y_reso + y_min
        
        grid = grid.view(2, -1).to(device) #[2, H * W]
        transformed_grid = pose[:2,:2] @ grid + pose[:2,3:4]
        
        # convert to [-1,1]
        transformed_grid[0] = transformed_grid[0] / abs(x_min)
        transformed_grid[1] = transformed_grid[1] / abs(y_min)
        
        transformed_grid = transformed_grid.view(2, H, W)
        
        return transformed_grid
            
    def warp_feats(self, bev_feats, pose_estimation):
        """warp features based on estimated transformation

        Args:
            bev_feats (tensor):         [B, T, C, Ny, Nx]
            pose_estimation (tensor):   [B, T, 4, 4], align any frame to the anchor frame
            coordinates (tensor):       [M, 5]
            points_mean (tensor):       [M, 3]
            
        Returns:
            updated_feats:              [B, T, C, Ny, Nx]
        """
        B, T, C, Ny, Nx = bev_feats.size()
        device = bev_feats.device
        
        updated_feats = []
        for batch_idx in range(B):
            transformed_grid_list = []
            for frame_idx in range(1, T):
                # align current frame to anchor frame
                c_pose = pose_estimation[batch_idx, frame_idx]  
                c_pose_inv = torch.linalg.inv(c_pose) # [4, 4]
                transformed_grid = self.get_transformed_grid(c_pose_inv, Ny, Nx, self.resolution[0], self.resolution[1], self.pc_range[0], self.pc_range[1])
                transformed_grid_list.append(transformed_grid)
            transformed_grids = torch.stack(transformed_grid_list).to(device) #[T-1, 2, H, W]
            transformed_grids = transformed_grids.permute(0,2,3,1)
            c_feats = bev_feats[batch_idx, 1:] 
            
            sampled_features = F.grid_sample(c_feats, transformed_grids, mode = 'bilinear', padding_mode = 'zeros', align_corners = False) #[T-1, C,, H, W]
            stacked_features = torch.cat((bev_feats[batch_idx, frame_idx:frame_idx+1], sampled_features), dim=0)
            updated_feats.append(stacked_features)
        updated_feats = torch.stack(updated_feats)
        return updated_feats   
    
    
    def transform_points(self, points, time_indice, transformation):
        """align point cloud to the same coordinate framework

        Args:
            points (tensor): [K, 3]
            time_indice (tensor): [K, 2]
            transformation (tensor): [B, T, 4,4]
        """
        B, T, _, _ = transformation.size()
        transformed_points = torch.ones_like(points)

        for batch_idx in range(B):
            for frame_idx in range(T):
                sel = ((time_indice[:,0] == batch_idx) & (time_indice[:,1] == frame_idx))
                c_points = points[sel]
                c_tsfm = transformation[batch_idx, frame_idx]
                c_transformed_points = ((c_tsfm[:3,:3] @ c_points.T) + c_tsfm[:3,3:4]).T
                transformed_points[sel] = c_transformed_points
        return transformed_points       

    def forward(self, input_dict):
        """Proposed motion net
        Args:
            input_dict (dict): input dictionary
            coordinates:    [M, 5] [batch_idx, z,y,x,t]
            num_points:     [M], avg = 3.24
            num_voxels:     [batch size] number of valid voxels in each batch
            point_to_voxel_map: [N]
        Returns:
            [type]: [description]
        """
        input_points = input_dict['input_points'].float() # [N, 3]
        time_indice = input_dict['time_indice'] #[N, 2]
        fb_labels = input_dict['fb_labels'] #[N, 1]
        point_to_voxel_map = input_dict['point_to_voxel_map'].long()[:,0] #[N]
        ego_motion_gt = input_dict['ego_motion_gt'].float() #[B, T, 4, 4]
        coordinates = input_dict["coordinates"] #[M, 5]
        num_voxels = input_dict["num_voxels"] #[B]
        input_shape = input_dict['shape'][0] #[B, 4]
        self.nt = input_shape[3]
        self.Nx = input_shape[0]
        self.Ny = input_shape[1]
        pillar_mean = scatter(input_points, point_to_voxel_map, dim=0, reduce='mean')  #[M, 3]
        fb_labels_sub = scatter(fb_labels, point_to_voxel_map, dim=0, reduce = 'max')   #[M, 1]
        assert pillar_mean.size(0) == coordinates.size(0)
        assert fb_labels_sub.size(0) == coordinates.size(0)
        batch_size = num_voxels.size(0)
        device = coordinates.device
        results = dict()    
    
        occupancy = torch.ones((coordinates.size(0),1)).to(device)
        occ_map = scatter_point_pillar(occupancy, coordinates, batch_size, input_shape).permute(0,2,1,3,4) #[B, T, 1, Ny, Nx]    
        fb_map = scatter_point_pillar(fb_labels_sub, coordinates, batch_size, input_shape).permute(0,2,1,3,4) #[B, T, 1, Ny, Nx]    
        pts_mean_map = scatter_point_pillar(pillar_mean, coordinates, batch_size, input_shape).permute(0,2,1,3,4) #[B,T,3,Ny, Nx]

        results['fb_seg_gt'] = fb_map
        results['occ_map'] = occ_map
        
        ###################################################
        # 1. pillar encoder
        input_features = self.pillar_encoder(input_points, point_to_voxel_map, coordinates, pillar_mean, time_indice)  #[M, 32]
        bev = scatter_point_pillar(input_features, coordinates, batch_size, input_shape) #[B, C, t, Ny, Nx]
        
        ###################################################
        # 2. unet backbone
        B, C, T, Ny, Nx = bev.size()
        bev = bev.permute(0,2,1,3,4).contiguous().view(B * T, C, Ny, Nx)  #[B*T, C, Ny, Nx]
        bev_feats = self.unet(bev)  #[B * T, C, Ny, Nx]

        ###################################################
        # 3. foreground background segmentation
        fb_seg = self.semseg_head(bev_feats) #[B * T, 2, Ny, Nx]
        fb_seg = fb_seg.view(B, T, 2, Ny, Nx)
        fb_est = fb_seg.max(dim=2, keepdim=True)[1] #[B, T, 1, Ny, Nx]
        results['fb_seg_est'] = fb_seg
        fb_est_occ_pillar = inverse_scatter_point_pillar(fb_est.permute(0,2,1,3,4).contiguous(), coordinates, batch_size, input_shape) #[M, C]
        fb_est_per_point = fb_est_occ_pillar[point_to_voxel_map] #[N, 1]
        results['fb_est_per_points'] = fb_est_per_point
        
        ###################################################
        # 4. project to geometric feature and run pairwise registration
        geometric_feats = self.ego_feats_head(bev_feats) #[B * T, C, Ny, Nx]
        geometric_feats = geometric_feats / torch.norm(geometric_feats, p=2, dim=1, keepdim=True)
        geometric_feats = geometric_feats.view(B, T, -1, Ny, Nx)  
        self.ego_motion_head(geometric_feats, fb_est, occ_map, pts_mean_map, ego_motion_gt, input_points, fb_est_per_point, time_indice, results)

        ###################################################
        # 5. motion segmentation
        pose_gt, pose_est = results['ego_motion_gt'], results['ego_motion_est'].float().detach()
        bev_feats = bev_feats.view(B, T, -1, Ny, Nx).detach() #[B, T, C, Ny, Nx]    
        warped_feats = self.warp_feats(bev_feats, pose_est)  #[B, T, C, H, W]
        warped_feats = warped_feats.permute(0, 2, 1, 3, 4)
        transformed_points = self.transform_points(input_points.clone(), time_indice, pose_est)
        results['transformed_points'] = transformed_points
        
        # we only decode the foreground points
        if self.mode in ['train','val']:    
            fb_mask = torch.logical_or(fb_labels[:,0] == 1, fb_est_per_point[:,0] == 1)
        else:
            fb_mask = fb_est_per_point[:,0] == 1

        full_mos = torch.zeros(transformed_points.size(0), 2).to(device)
        full_offset = torch.zeros(transformed_points.size(0), 2).to(device)
        full_mos[:,0] = 1
        
        if fb_mask.sum() > MIN_POINTS:
            mos, offset, mos_feats = self.motionhead(warped_feats, transformed_points.clone()[fb_mask], time_indice[fb_mask], self.pc_range)
            full_mos[fb_mask] = mos
            full_offset[fb_mask] = offset

        results['mos_est'] = full_mos
        results['offset_est'] = full_offset
        results['rec_est'] = transformed_points.clone()

        ###################################################
        # 6. TubeNet
        reconstructor_input = dict()
        if self.mode in ['train','val']:    
            inst_labels = input_dict['inst_labels'][:,0].long()
            rec_mask = input_dict['fb_labels'][:,0] == 1 
        else:
            self.cluster(transformed_points, full_mos.argmax(1), full_offset, time_indice, results, use_offset = True)
            
            inst_labels = results['inst_labels_est']
            rec_mask = inst_labels!= 0 
            
        if rec_mask.sum()>MIN_POINTS:
            backbone_feats = temporal_ungrid(bev_feats, input_points[rec_mask].clone(), self.pc_range, time_indice[rec_mask])
            motion_feats = ungrid(mos_feats, transformed_points[rec_mask].clone(), self.pc_range, time_indice[rec_mask])
            
            reconstructor_input = {
                    'inst_labels': inst_labels[rec_mask],  
                    'time_indice': time_indice[rec_mask],
                    'transformed_points': transformed_points[rec_mask],
                    'backbone_feats': backbone_feats,
                    'motion_feats': motion_feats,
                    'inst_motion_gt': input_dict['inst_motion_gt'],
                    'mos_labels': input_dict['sd_labels'][rec_mask,0].long(),
                    'ego_motion_est': results['ego_motion_est'],
                    'ego_motion_gt': results['ego_motion_gt']
                } 
                
            self.reconstructor(reconstructor_input, results)    
            results['rec_est'][rec_mask] = results['sub_rec_est']          

        return results