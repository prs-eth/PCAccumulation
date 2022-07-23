import torch.nn as nn
import torch
from models.unet import DownConv, UpConv, SegHead1D
from models.pillar_encoder import ungrid
    

class STPN(nn.Module):
    def __init__(self, height_feat_size=32):
        super(STPN, self).__init__()
        n_filters = [32, 64, 128, 128, 256]
        
        # initial 3D convolution to aggregation temporal information
        self.init_conv = nn.Sequential(
            nn.Conv3d(height_feat_size, n_filters[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(n_filters[0], n_filters[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(n_filters[0], n_filters[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(n_filters[0], n_filters[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        # small unet to aggregation global context
        depth = len(n_filters)
        self.down_convs = []
        ins = height_feat_size
        for idx, width in enumerate(n_filters):
            pooling = True if idx < depth - 1 else False
            width = max(64, width)
            self.down_convs.append(DownConv(ins, width, pooling = pooling))
            ins = width
        
        self.up_convs = []
        ins = height_feat_size
        ins = n_filters[-1]
        for idx, width in enumerate(n_filters[-2::-1]):
            width = max(64, width)
            up_conv = UpConv(ins, width, merge_mode ='concat')
            self.up_convs.append(up_conv)
            ins = width
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
            
        # per_point positional encoding
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.ReLU()
        )
            
        # motion segmentation head
        self.mos_seg = SegHead1D(128 , 2)
        self.offset_head = SegHead1D(128, 2)        
            
    def safe_guard_offset(self, offset, min = -20, max = 20):
        offset = torch.where(torch.isnan(offset), torch.zeros_like(offset), offset)
        offset = torch.where(torch.isinf(offset), torch.zeros_like(offset), offset)
        offset = torch.clamp(offset, min, max)
        return offset

    def forward(self, x, points, time_indice, pc_range):
        """motion segmentation head
        Here for final decoding, we only operate over the foreground parts
        Args:
            x (tensor):             [B, C, T, H, W]
            points (tensor):        [K, 3]]
            time_indice (tensor):   [K, 2]]

        Returns:
            classes (tensor):       [K, 2]
            offset  (tensor):       [K, 2]
        """
        x = self.init_conv(x)
        x = torch.max(x, dim=2)[0]  #[B, C, H, W]
        
        # unet to get per-pillar feature
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        ungridded_feats = ungrid(x, points.clone(), pc_range, time_indice) # [K, C]
        
        # positional encoding 
        pos_input = points / abs(pc_range[0])
        pos_encoding = self.positional_encoding(pos_input)
        
        # add the features
        final_encoding = torch.cat([pos_encoding, ungridded_feats], dim=-1)
        final_encoding = self.final_proj(final_encoding)
        
        classes = self.mos_seg(final_encoding)
        offset = self.offset_head(final_encoding)
        offset = self.safe_guard_offset(offset)

        return classes, offset, x