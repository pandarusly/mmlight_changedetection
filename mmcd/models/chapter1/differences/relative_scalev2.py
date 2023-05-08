from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import BaseModule

# from ..builder import ATTENTION, DIFFERENCE, build_attn


# with_cross_attention
# @ATTENTION.register_module()
class RelativeAttention(nn.Module):
    def __init__(self, dim,layer):
        """ A | B -> ( A*C | B *C ) * L  -> A | B
        """
        super().__init__()
        # time_digger_block = 
        self.extract_time_info = nn.Conv3d(dim,dim,(2,3,3),padding=(0,1,1))
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        )

    def forward(self, x, y):
        u = x.clone()
        v = y.clone()
        
        # ------------------- 收集为一个类。具备用于比较卷积，代数求和，欧式距离，度量。
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.extract_time_info(attn).squeeze(2)  # b c h w
        # ------------------- 

        u = u * attn
        u = self.conv1(u)

        v = v * attn
        v = self.conv1(v)

        return u,v

# class RelativeFormer(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.layers = nn.ModuleList(*)





# --self_attention
# @ATTENTION.register_module()
class DenseFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        out = x1 + x2 + x3
        return out


# @DIFFERENCE.register_module()
class SiameseRelativeScale(BaseModule):
    def __init__(self, cross_attention_cfg=None, init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.cross_attention12 = build_attn(
            cross_attention_cfg) if cross_attention_cfg is not None else None

        self.cross_attention21 = build_attn(
            cross_attention_cfg) if cross_attention_cfg is not None else None

        self.scale_12 = Scale(0)
        self.scale_21 = Scale(0)

    @property
    def with_self_attention(self):
        return hasattr(self, 'self_attention') and self.self_attention is not None

    @property
    def with_cross_attention(self):
        return hasattr(self, 'cross_attention') and self.cross_attention is not None


    def forward(self,t1, t2):

        Across1 = self.scale_12(self.cross_attention12(t1, t2))
        Across2 = self.scale_21(self.cross_attention21(t2, t1))

        attn_1 = t1 + Across1
        attn_2 = t2 + Across2

        return torch.abs(attn_1-attn_2)


if __name__ == '__main__':
    import torch

    def cal_params(model):
        import numpy as np
        p = filter(lambda p: p.requires_grad, model.parameters())
        p = sum([np.prod(p_.size()) for p_ in p]) / 1_000_000
        print('%.3fM' % p)

    in_channels = [32, 64, 160, 256]
    # [128//2, 256//2, 512//2, 1024//2]
    scales = [340, 170, 84, 43]
    in_channels = [128]
    scales = [256]
    a = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    b = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]

    # fuse = FusionHEAD(in_channels=2, channels=4, fusion_form='attn', act_cfg=None, norm_cfg=None, dilations=[1, 4])
    fuse = SiameseRelativeScale(
        DenseFlow(128),
        PDA(128),
    )

    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
