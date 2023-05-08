
import torch

import torch.nn as nn

from ..builder import ATTENTION

@ATTENTION.register_module()
class LAKForChange(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.proj2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        )

    def forward(self, x, y): 
        u = x.clone()
        v = y.clone()
        attn = torch.cat([x, y], dim=1)  # b 2c h w
        attn = self.proj1(attn)

        attn = self.conv0(attn)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        u = u * attn
        u = self.proj2(u)
        v = v * attn
        v = self.proj2(v)
        return u, v
    
 
@ATTENTION.register_module()
class LAKForSegmentation(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

  
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x ): 
        u = x.clone()   

        attn = self.conv0(u)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        u = u * attn 
        return u 
 