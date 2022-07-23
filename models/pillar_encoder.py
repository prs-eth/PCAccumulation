"""
Code written by Alex Lang and Oscar Beijbom, 2018
     modified by Shengyu Huang, 2022
Licensed under MIT License [see LICENSE].
"""
from torch_scatter import scatter
import torch
from torch import nn
from torch.nn import functional as F


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx



class PillarFeatureNet(nn.Module):
    def __init__(self, cfg):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super(PillarFeatureNet, self).__init__()
        num_input_features = cfg['num_input_features']
        num_filters = cfg['num_filters']
        voxel_size = cfg['voxel_size']
        pc_range = cfg['pc_range']
        depth = cfg['depth']
        
        self.scale = abs(pc_range[0])
        self.n_frames = cfg['n_sweeps']
        self.name = "PillarFeatureNet"
        #modify
        
        self.fc_pos = nn.Linear(num_input_features, 2 * num_filters)
        self.fc_c = nn.Linear(num_filters, num_filters)
        self.actvn = nn.ReLU()
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*num_filters, num_filters) for i in range(depth)
        ])
        

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, raw_points, point_to_voxel_map, coordinates, pillar_mean, time_indice):
        dist_to_pts_mean = raw_points - pillar_mean[point_to_voxel_map]
        f_center = torch.zeros_like(raw_points[:,:2])
        
        mapped_coords = coordinates[point_to_voxel_map]
        
        f_center[:,0] = raw_points[:,0] - (mapped_coords[:,3] * self.vx  + self.x_offset)
        f_center[:,1] = raw_points[:,1] - (mapped_coords[:,2] * self.vy + self.y_offset)

        # normalise the input
        features = [raw_points, dist_to_pts_mean, f_center, time_indice[:,1:2]]
        features = torch.cat(features, dim=-1).float()
        features[:,:-1] /= self.scale
        features[:,-1] /= self.n_frames
        
        # forward pass
        net = self.fc_pos(features)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = scatter(net, point_to_voxel_map, dim=0, reduce = 'max')[point_to_voxel_map]
            net = torch.cat([net, pooled], dim=1)
            net = block(net)
        feats = self.fc_c(net)
        feats = scatter(feats, point_to_voxel_map, dim=0, reduce ='max')
        
        return feats
    
    
def scatter_point_pillar(voxel_features, coords, batch_size, input_shape):
    """Scatter the pillar features to Pseudo image

    Args:
        voxel_features (tensor):      [M, C]      
        coords (tensor):              [M, 5]
        batch_size (tensor):          integer          
        input_shape (tensor):         [nx, ny, nz, nt]

    Returns:
        batch_canvas (tensor):        [B, C, nt, ny, nx]
    """
    n_channels = voxel_features.size(1)   
    nt = input_shape[3]
    nx = input_shape[0]
    ny = input_shape[1]
    
    # batch_canvas will be the final output.
    batch_canvas = []
    for batch_itt in range(batch_size):
        # Create the canvas for this sample
        canvas = torch.zeros(
            n_channels,
            nt * nx * ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )

        # Only include non-empty pillars
        batch_mask = coords[:, 0] == batch_itt
        
        this_coords = coords[batch_mask, :]
        
        indices = this_coords[:, 4] * nx * ny + this_coords[:, 2] * nx + this_coords[:, 3]
        indices = indices.type(torch.long)
        voxels = voxel_features[batch_mask, :]
        voxels = voxels.t()

        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels

        # Append to a list for later stacking.
        batch_canvas.append(canvas)

    # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
    batch_canvas = torch.stack(batch_canvas, 0)

    # Undo the column stacking to final 4-dim tensor
    batch_canvas = batch_canvas.view(batch_size, n_channels, nt, ny, nx)
    return batch_canvas


def inverse_scatter_point_pillar(voxel_features, coords, batch_size, input_shape):
    """Inverse the scatter proceedure

    Args:
        voxel_features (_type_):        [n, c, nt, ny, nx]
        coords (_type_): _description_
        batch_size (_type_): _description_
        input_shape (_type_): _description_
    """
    n_channels = voxel_features.size(1)   
    nt = input_shape[3]
    nx = input_shape[0]
    ny = input_shape[1]
    
    # batch_canvas will be the final output.
    batch_feats_list = []
    for batch_itt in range(batch_size):
        # Only include non-empty pillars
        batch_mask = coords[:, 0] == batch_itt
        this_coords = coords[batch_mask, :]
        
        indices = this_coords[:, 4] * nx * ny + this_coords[:, 2] * nx + this_coords[:, 3]
        indices = indices.type(torch.long)
        
        this_feats = voxel_features[batch_itt].view(n_channels,-1)[:,indices].t()
        batch_feats_list.append(this_feats)
    batch_feats = torch.cat(batch_feats_list)
    return batch_feats

def temporal_ungrid(feats, points, pc_range, time_indice):
    """Bilinear interpolation to retrive features for each point

    Args:
        feats (tensor):         [B, T, C, H, W]
        points (tensor):        [N, 3]
        pc_range (tensor):      [2]
        time_indice (tensor):   [N, 2] [batch_idx, time_indice]

    Returns:
       ungridded_feats:         [N, C]
    """
    B, T, C, H, W = feats.size()
    n_points = points.size(0)
    device = feats.device
    ungridded_feats= torch.zeros(n_points, C).to(device)
    for time_idx in range(T):
        sel = time_indice[:,1] == time_idx
        if sel.sum():
            temporal_feats, temporal_points = feats[:,time_idx], points[sel]
            ungridded_feats[sel] = ungrid(temporal_feats, temporal_points, pc_range, time_indice[sel])
    
    return ungridded_feats


def ungrid(feats, points, pc_range, time_indice):
    """Bilinear interpolation to retrive features for each point

    Args:
        feats (tensor):         [B, C, H, W]
        points (tensor):        [N, 3]
        pc_range (tensor):      [2]
        time_indice (tensor):   [N, 2] [batch_idx, time_indice]

    Returns:
       ungridded_feats:         [N, C]
    """
    B, C, H, W = feats.size()
    count = H * W
    
    # convert points to [-1, 1] grid
    uv = points[:,:2]
    uv[:,0] = uv[:,0] / abs(pc_range[0])
    uv[:,1] = uv[:,1] / abs(pc_range[1])
    
    ungridded_feats_list = []
    for batch_idx in range(B):
        sel = time_indice[:,0] == batch_idx
        c_uv = uv[sel] # [K, 2]
        
        n, res = c_uv.size(0) // count, c_uv.size(0) % count
        n_padding = count - res
        padded_zero = torch.zeros((n_padding, 2)).to(c_uv.device)
        c_uv = torch.cat((c_uv, padded_zero), dim=0).contiguous().view(n+1,H, W, 2)
        c_feats_map = feats[batch_idx:batch_idx+1].repeat(n+1, 1,1,1)
        ungridded_feats = F.grid_sample(c_feats_map, c_uv, mode = 'bilinear', padding_mode = 'border', align_corners = False) #[n+1, C,, H, W]
        
        ungridded_feats = ungridded_feats.permute(0,2,3,1).contiguous().view(-1, C)
        ungridded_feats = ungridded_feats[:n*count+res, :]
        ungridded_feats_list.append(ungridded_feats)
    ungridded_feats = torch.cat(ungridded_feats_list, dim=0)
    return ungridded_feats     