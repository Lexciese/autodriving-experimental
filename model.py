import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


# --------------------------------------------------------------
# 1. Weight Initialization
# --------------------------------------------------------------
def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


# --------------------------------------------------------------
# 2. Basic Building Blocks
# --------------------------------------------------------------
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final=False):
        super().__init__()
        if final:
            self.block = nn.Sequential(
                ConvBNRelu(in_channels, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.block = nn.Sequential(
                ConvBNRelu(in_channels, out_channels),
                ConvBNRelu(out_channels, out_channels)
            )
        self.apply(kaiming_init)

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------------
# 3. Main Model
# --------------------------------------------------------------
class ai23(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.gpu_device = device

        # ------------------- A. RGB Encoder -------------------
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b3(pretrained=True)
        
        # Hanya kunci 2 layer terawal, biarkan fitur mid-high level belajar domain driving
        for i in range(2): 
            for param in self.RGB_encoder.features[i].parameters():
                param.requires_grad = False

        self.RGB_encoder.classifier = nn.Identity()
        self.RGB_encoder.avgpool = nn.Identity()

        # ------------------- B. Segmentation Decoder -------------------
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv3_ss_f = ConvBlock(self.config.n_fmap_b3[4][-1] + self.config.n_fmap_b3[3][-1], self.config.n_fmap_b3[3][-1])
        self.conv2_ss_f = ConvBlock(self.config.n_fmap_b3[3][-1] + self.config.n_fmap_b3[2][-1], self.config.n_fmap_b3[2][-1])
        self.conv1_ss_f = ConvBlock(self.config.n_fmap_b3[2][-1] + self.config.n_fmap_b3[1][-1], self.config.n_fmap_b3[1][-1])
        self.conv0_ss_f = ConvBlock(self.config.n_fmap_b3[1][-1] + self.config.n_fmap_b3[0][-1], self.config.n_fmap_b3[0][0])
        self.final_ss_f = ConvBlock(self.config.n_fmap_b3[0][0], self.config.n_class, final=True)

        # ------------------- C. Semantic Cloud Projection -------------------
        self.cover_area = self.config.coverage_area
        self.h = int(self.config.crop_roi[0] / self.config.scale)
        self.w = int(self.config.crop_roi[1] / self.config.scale)
        
        self.SC_encoder = models.efficientnet_b1(pretrained=False)
        self.SC_encoder.features[0][0] = nn.Conv2d(self.config.n_class, self.config.n_fmap_b1[0][0],
                                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.SC_encoder.classifier = nn.Identity()
        self.SC_encoder.avgpool = nn.Identity()
        self.SC_encoder.apply(kaiming_init)

        # ------------------- D. Fusion & Sequence Modeller -------------------
        self.necks_net = nn.Sequential(
            nn.Conv2d(self.config.n_fmap_b3[4][-1] + self.config.n_fmap_b1[4][-1],
                      self.config.n_fmap_b3[4][1], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.config.n_fmap_b3[4][1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(self.config.n_fmap_b3[4][1], self.config.n_fmap_b3[4][0])
        )

        self.gru = nn.GRUCell(input_size=7, hidden_size=self.config.n_fmap_b3[4][0])
        self.pred_dwp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.config.n_fmap_b3[4][0], 2)
        )

    # --------------------------------------------------------------
    # 4. Differentiable Semantic Cloud Generator
    # --------------------------------------------------------------
    def gen_top_view_sc_ptcloud(self, pt_cloud_x, pt_cloud_z, semseg_logits):
        B, C, H, W = semseg_logits.shape
        device = semseg_logits.device
        
        semseg_probs = F.softmax(semseg_logits, dim=1) 
        top_view_sc = torch.zeros((B, C, H, W), device=device, dtype=semseg_probs.dtype)
        
        px = torch.round((pt_cloud_x + self.cover_area) * (W - 1) / (2 * self.cover_area)).long() 
        pz = torch.round((pt_cloud_z * (1 - H) / self.cover_area) + (H - 1)).long()               
        
        valid_mask = (px >= 0) & (px < W) & (pz >= 0) & (pz < H) 
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, 1, H, W)
        
        v_b = batch_idx[valid_mask]
        v_px = px[valid_mask]
        v_pz = pz[valid_mask]
        
        for c in range(C):
            v_probs = semseg_probs[:, c:c+1, :, :][valid_mask]
            top_view_sc.index_put_((v_b, torch.full_like(v_b, c), v_pz, v_px), v_probs, accumulate=True)
            
        top_view_sc = torch.clamp(top_view_sc, 0.0, 1.0)
        return top_view_sc

    # --------------------------------------------------------------
    # 5. Batch-Vectorized Forward Pass
    # --------------------------------------------------------------
    def forward(self, rgbs_list, pt_cloud_xs_list, pt_cloud_zs_list, rp1, rp2, velo_in):
        B = rgbs_list[0].size(0)
        S = self.config.seq_len
        
        rgbs_stacked = torch.cat(rgbs_list, dim=0) 
        pt_cloud_xs_stacked = torch.cat(pt_cloud_xs_list, dim=0)
        pt_cloud_zs_stacked = torch.cat(pt_cloud_zs_list, dim=0)

        in_rgb = self.rgb_normalizer(rgbs_stacked)
        f0 = self.RGB_encoder.features[0](in_rgb)
        f1 = self.RGB_encoder.features[1](f0)
        f2 = self.RGB_encoder.features[2](f1)
        f3 = self.RGB_encoder.features[3](f2)
        f4 = self.RGB_encoder.features[4](f3)
        f5 = self.RGB_encoder.features[5](f4)
        f6 = self.RGB_encoder.features[6](f5)
        f7 = self.RGB_encoder.features[7](f6)
        f8 = self.RGB_encoder.features[8](f7)

        ss_f_3 = self.conv3_ss_f(torch.cat([self.up(f8), f5], dim=1))
        ss_f_2 = self.conv2_ss_f(torch.cat([self.up(ss_f_3), f3], dim=1))
        ss_f_1 = self.conv1_ss_f(torch.cat([self.up(ss_f_2), f2], dim=1))
        ss_f_0 = self.conv0_ss_f(torch.cat([self.up(ss_f_1), f1], dim=1))
        segs_logits_stacked = self.final_ss_f(self.up(ss_f_0)) 
        
        segs_f = list(torch.split(segs_logits_stacked, B, dim=0))

        sdcs_stacked = self.gen_top_view_sc_ptcloud(pt_cloud_xs_stacked, pt_cloud_zs_stacked, segs_logits_stacked)
        sdcs = list(torch.split(sdcs_stacked, B, dim=0)) 

        sc0 = self.SC_encoder.features[0](sdcs_stacked)
        sc1 = self.SC_encoder.features[1](sc0)
        sc2 = self.SC_encoder.features[2](sc1)
        sc3 = self.SC_encoder.features[3](sc2)
        sc4 = self.SC_encoder.features[4](sc3)
        sc5 = self.SC_encoder.features[5](sc4)
        sc6 = self.SC_encoder.features[6](sc5)
        sc7 = self.SC_encoder.features[7](sc6)
        sc8 = self.SC_encoder.features[8](sc7) 

        RGB_features_sum = f8.view(S, B, f8.size(1), f8.size(2), f8.size(3)).sum(dim=0)
        SC_features_sum = sc8.view(S, B, sc8.size(1), sc8.size(2), sc8.size(3)).sum(dim=0)

        hx = self.necks_net(torch.cat([RGB_features_sum, SC_features_sum], dim=1)) 

        xy = torch.zeros((B, 2), device=self.gpu_device, dtype=hx.dtype)
        
        velo_in = torch.clamp(velo_in, 0, 30) / 30.0   
        velo_in = velo_in.unsqueeze(1)
        rp1 = rp1 / 50.0    
        rp2 = rp2 / 50.0

        out_wp = []
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1) 
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx) 
            xy = xy + d_xy
            out_wp.append(xy)

        pred_wp = torch.stack(out_wp, dim=1) 
        pred_wp_meter = pred_wp * 50.0 

        return segs_f, pred_wp_meter, sdcs
