import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from torch.nn.modules.conv import Conv2d
import torch.nn.functional as F

from ..builder import DIFFERENCE,ATTENTION,build_attention


@DIFFERENCE.register_module()
class LKADIFF(nn.Module):
    def __init__(self, dim, k=5, td=True, k2=3) -> None:
        super().__init__()
        pad = k // 2
        self.td = td
        if td:
            self.conv0 = nn.Conv3d(
                dim, dim,
                kernel_size=(2, k, k),
                padding=(0, pad, pad),
                groups=dim
            )
        else:
            self.conv0 = nn.Conv2d(
                dim, dim,
                kernel_size=k,
                padding=pad,
                groups=dim
            )

        self.point_wise = Conv2d(
            dim, dim, kernel_size=1
        )

        if k2 != 0:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k2, padding=k2 // 2),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Identity()

    def forward_ed(self, x, y):
        u = x.clone()
        x = self.conv0(x)
        x = self.point_wise(x)
        u = u * x
        u = self.conv1(u)

        v = y.clone()
        y = self.conv0(y)
        y = self.point_wise(y)
        v = v * y
        v = self.conv1(v)
        return u, v

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.conv0(attn).squeeze(2)  # b c h w
        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward(self, x, y):
        if self.td:
            x, y = self.forward_td(x, y)
        else:
            x, y = self.forward_ed(x, y)

        return torch.abs(x - y)


@DIFFERENCE.register_module()
class LKADIFFV2(nn.Module):
    def __init__(self, dim, **kwargs) -> None:
        super().__init__()

        # self.conv0 = nn.Conv3d(
        #     dim, dim,
        #     kernel_size=5,
        #     padding=5//2,
        #     groups=dim
        # )

        self.conv0 = nn.Conv3d(
            dim, dim,
            kernel_size=(2, 5, 5),
            padding=(0, 5//2, 5//2),
            groups=dim
        )

        # self.conv0_1 = nn.Conv3d(dim, dim, (2, 1, 7), padding=(0, 0, 3), groups=dim)
        # self.conv0_2 = nn.Conv3d(dim, dim, (2, 7, 1), padding=(0, 3, 0), groups=dim)

        # self.conv1_1 = nn.Conv3d(dim, dim, (2, 1, 11), padding=(0,0, 5), groups=dim)
        # self.conv1_2 = nn.Conv3d(dim, dim, (2, 11, 1), padding=(0,5, 0), groups=dim)

        # self.conv2_1 = nn.Conv3d(
        #     dim, dim, (2, 1, 21), padding=(0,0, 10), groups=dim)
        # self.conv2_2 = nn.Conv3d(
        #     dim, dim, (2, 21, 1), padding=(0,10, 0), groups=dim)

        # self.point_wise = Conv2d(
        #     dim, dim, kernel_size=1
        # )

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.point_wise = Conv2d(
            dim, dim, kernel_size=1
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        )

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.conv0(attn).squeeze(2)  # b c h w #

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward(self, x, y):

        x, y = self.forward_td(x, y)

        return torch.abs(x - y)


@DIFFERENCE.register_module()
class LKADIFFV3(nn.Module):
    def __init__(self, dim, **kwargs) -> None:
        super().__init__()

        self.conv0 = nn.Conv3d(
            dim, dim,
            kernel_size=(2, 5, 5),
            padding=(0, 5//2, 5//2),
            groups=dim
        )

        self.conv0_1 = nn.Conv3d(dim, dim, (3, 1, 7), padding=(1, 0, 3), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (3, 7, 1), padding=(1, 3, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (3, 1, 11), padding=(1,0, 5), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (3, 11, 1), padding=(1,5, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (3, 1, 21), padding=(1,0, 10), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (3, 21, 1), padding=(1,10, 0), groups=dim)

        self.point_wise = Conv2d(
            dim, dim, kernel_size=1
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        )

    def forward_td(self, x, y):
        u = x.clone()
        v = y.clone()
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.conv0(attn) 

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2 # b c t h w

        attn = self.point_wise(attn)
        u = u * attn
        u = self.conv1(u)
        v = v * attn
        v = self.conv1(v)
        return u, v

    def forward(self, x, y):

        x, y = self.forward_td(x, y)

        return torch.abs(x - y)



# ------------------ 
@ATTENTION.register_module()
class LKA(nn.Module):
    def __init__(self, dim, k=5) -> None:
        super().__init__()
        pad = k // 2 
 
        self.conv0 = nn.Conv3d(
            dim, dim,
            kernel_size=(2, k, k),
            padding=(0, pad, pad),
            groups=dim
        )

        self.point_wise = Conv2d(
            dim, dim, kernel_size=1
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        )

    def forward(self, x, y): 
        u = x.clone()
        v = y.clone()
        attn = torch.stack([x, y], dim=2)  # b c t h w
        attn = self.conv0(attn).squeeze(2)  # b c h w
        attn = self.point_wise(attn)
        u = u * attn
        u = self.proj2(u)
        v = v * attn
        v = self.proj2(v)
        return  torch.abs(u - v)
    


@DIFFERENCE.register_module()
class abs_diff(nn.Module):
    def __init__(self,i_c,o_c=None,num_convs=1,attn_cfg=None,**kwargs):
        super().__init__()
        self.i_c = i_c
        self.o_c = o_c or i_c 
        self.attn = build_attention(
            attn_cfg) if attn_cfg is not None else None
        
        if num_convs == 0:
            assert self.o_c == i_c
        convs = []
        for i in range(num_convs):
            _in_channels = self.i_c if i == 0 else self.o_c
            convs.append(
            torch.nn.Sequential(
            torch.nn.Conv2d(_in_channels, self.o_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.o_c),
            torch.nn.ReLU(inplace=True),
        ))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

    def forward(self, x, y):
        if self.attn:
            x,y = self.attn(x,y)
        return self.convs(torch.abs(x - y))

@DIFFERENCE.register_module()
class sum_diff(nn.Module):
    def __init__(self,i_c,o_c=None,num_convs=1,attn_cfg=None,**kwargs):
        super().__init__()
        self.i_c = i_c
        self.o_c = o_c or i_c
        self.attn = build_attention(
            attn_cfg) if attn_cfg is not None else None
        
        if num_convs == 0:
            assert self.o_c == i_c

        convs = []
        for i in range(num_convs):
            _in_channels = self.i_c if i == 0 else self.o_c
            convs.append(
            torch.nn.Sequential(
            torch.nn.Conv2d(_in_channels, self.o_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.o_c),
            torch.nn.ReLU(inplace=True),
        ))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

    def forward(self, x, y):
        if self.attn:
            x,y = self.attn(x,y)
        return self.convs(torch.sum(x , y))


@DIFFERENCE.register_module()
class conv_diff(nn.Module):
    def __init__(self,i_c,o_c=None,num_convs=1,attn_cfg=None,**kwargs):
        super().__init__()
        self.i_c = i_c*2
        self.o_c = o_c or i_c
        self.attn = build_attention(
            attn_cfg) if attn_cfg is not None else None
        
        if num_convs == 0:
            assert self.o_c == i_c

        convs = []
        for i in range(num_convs):
            _in_channels = self.i_c if i == 0 else self.o_c
            convs.append(
            torch.nn.Sequential(
            torch.nn.Conv2d(_in_channels, self.o_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.o_c),
            torch.nn.ReLU(inplace=True),
        ))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

    def forward(self, x, y):
        if self.attn:
            x,y = self.attn(x,y)
        _fuse1 = torch.cat([x,y],dim=1)
        _fuse2 = torch.cat([y,x],dim=1)
        return self.convs(_fuse1+_fuse2)


def torch10_pairwise_distance(x1,x2):
    x1 = x1.permute(0,2,3,1)
    x2 = x2.permute(0,2,3,1)
    x = F.pairwise_distance(x1,x2,keepdim=True)
    x= x.permute(0,3,1,2)
    return x 

@DIFFERENCE.register_module()
class metric_diff(nn.Module):
    def __init__(self,torch10=False,attn_cfg=None,**kwargs):
        super().__init__() 
        self.torch10 = torch10

        self.attn = build_attention(
            attn_cfg) if attn_cfg is not None else None
            
    def forward(self, x, y):
        if self.attn:
            x,y = self.attn(x,y)
        
        dist = torch10_pairwise_distance(x,y)  if self.torch10 else F.pairwise_distance(x,y, keepdim=True) 
            # channel = 1
        return dist

# -----------------dense_block_3d


class dense_block_3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        return x1 + x2 + x3


@DIFFERENCE.register_module()
class attn_fusion(nn.Module):
    def __init__(self, in_chn=128, out_chn=128, r=8):
        super().__init__()

        self.attn = nn.Sequential(
            dense_block_3d(out_chn),  # 3
            Rearrange("b c t h w -> b (c t) h w"),  # 4
            nn.Conv2d(in_chn * 2, out_chn, kernel_size=1, padding=0),  # 5
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x_3d = torch.stack([x, y], dim=2)
        return self.attn(x_3d)


# ------------------- dense_diff
@DIFFERENCE.register_module()
class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()

        if in_chn != out_chn:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_chn),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.reduction = torch.nn.Identity()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, in_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.reduction(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y = self.reduction(y)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


# ------------------- dw_diff


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        kernel_size (int):  e. Default: 7.
    """

    def __init__(self, dim, kernel_size=7, scale=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, int(scale * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(scale * dim), dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, drop_path=0., kernel_size=7, scale=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, int(scale * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(scale * dim), dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


@DIFFERENCE.register_module()
class depwise_diff(nn.Module):
    def __init__(self, in_chn=128, out_chn=128, scale=0.5, kernel_size=7, drop_path_rate=0.0, depth=0):

        super().__init__()

        if in_chn != out_chn:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_chn),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.reduction = torch.nn.Identity()

        convs = []
        # drop_path_rate = drop_path_rate
        convs.append(
            nn.Sequential(
                Block(out_chn, kernel_size=kernel_size, scale=scale)
            )
        )

        dp_rates = [x.item() for x in torch.linspace(
            0, drop_path_rate, depth - 1)] if depth != 0 else [0.0]

        for i in range(depth - 1):
            convs.append(
                nn.Sequential(
                    ResBlock(out_chn, kernel_size=kernel_size,
                             drop_path=dp_rates[i], scale=scale)
                )
            )

        if depth == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, in_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.reduction(x)
        x1 = self.convs(x)

        y = self.reduction(y)
        y1 = self.convs(y)
        out = self.conv_out(torch.abs(x1 - y1))
        return out


class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


# class densecat_cat_diff(nn.Module):
#     def __init__(self, in_chn, out_chn):
#         super(densecat_cat_diff, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv_out = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_chn),
#             torch.nn.ReLU(inplace=True),
#         )

#     def forward(self, x, y):

#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2+x1)

#         y1 = self.conv1(y)
#         y2 = self.conv2(y1)
#         y3 = self.conv3(y2+y1)
#         out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
#         return out


@DIFFERENCE.register_module()
class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


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
    fuse = LKADIFFS(128, [13, 3, 5, 7])

    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
    fuse = LKADIFF(128, 13, True)
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
    fuse = LKADIFFSV2(128, [13, 3, 5, 7])
    cal_params(fuse)
    x = fuse(a[0], b[0])
    print(x.shape)
