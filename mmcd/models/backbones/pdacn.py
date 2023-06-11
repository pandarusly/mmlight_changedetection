# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES
from mmcv.cnn.utils.weight_init import constant_init, normal_init, trunc_normal_init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from mmseg.ops import Upsample,resize
import warnings
# ----------------------
from mmcv.utils import Registry 
SEGHEADS = Registry('seghead')
CHANGES = Registry('change')
 

def build_seghead(cfg):

    return SEGHEADS.build(cfg)
def build_change(cfg):

    return CHANGES.build(cfg)


 
# -----------------SEGHEADS
@SEGHEADS.register_module()
class SimpleFuse(BaseModule):

    def __init__(
        self,
        in_channels=[32,64,160,256],
        channels=128,
        init_cfg=dict(type="Normal", std=0.01),
        ):
        super(SimpleFuse, self).__init__(init_cfg)

        self.channels = channels
        self.in_channels = in_channels
        
        num_inputs = len(self.in_channels)

        self.convs = nn.ModuleList()

        for i in range(num_inputs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i],self.channels,1,1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                ))

        self.fusion_conv = nn.Sequential(
                    nn.Conv2d(self.channels * num_inputs,self.channels,1,1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        return out

# ----------------------

@CHANGES.register_module()
class LKADIFF(nn.Module):
    def __init__(self,dim,k,td=False,k2=3) -> None:
        super().__init__()
        pad = k //2
        self.td = td
        if td:
            self.conv0 = nn.Conv3d(
                dim,dim,
                kernel_size=(2,k,k),
                padding=(0,pad,pad),
                groups=dim
            )
        else:
            self.conv0 = nn.Conv2d(
                dim,dim,
                kernel_size=k,
                padding=pad,
                groups=dim
            )


        self.point_wise = nn.Conv2d(
            dim,dim,kernel_size=1
        )

        if k2 !=0:
            self.conv1=nn.Sequential(
                nn.Conv2d(dim,dim,kernel_size=k2,padding=k2//2),
                nn.ReLU()
            )
        else: 
            self.conv1=nn.Identity()

    def forward_ed(self,x,y):
        u = x.clone()
        x = self.conv0(x)
        x = self.point_wise(x)
        u = u*x
        u = self.conv1(u)

        v = y.clone()
        y = self.conv0(y)
        y = self.point_wise(y)
        v = v*y
        v = self.conv1(v)
        return u,v

    def forward_td(self, x,y):
        u = x.clone()
        v = y.clone()
        attn = torch.stack([x,y],dim=2) # b c t h w 
        attn = self.conv0(attn).squeeze(2) # b c h w 
        attn = self.point_wise(attn)
        u = u*attn
        u = self.conv1(u)
        v = v*attn
        v = self.conv1(v)
        return u,v

    def forward(self,x,y):
        if self.td:
            x,y  = self.forward_td(x,y)
        else:
            x,y  = self.forward_ed(x,y)

        return torch.abs(x-y)



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

        return x1+x2+x3

# -----------------dense_block_3d

@CHANGES.register_module()
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
@CHANGES.register_module()
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
            self.reduction =  torch.nn.Identity()

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

# ------------------
@CHANGES.register_module()
class fcn(BaseModule):
    def __init__(self,
        in_channels=128,
        channels=128,
        num_convs=2,
        kernel_size=3,
        num_classes=2,
        concat_input=True,
        dilation=1,
        dropout_ratio=0.1,
        init_cfg=None,
        upsample_sacle=4,
        ):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.in_channels = in_channels 
        self.channels = channels
        super().__init__(init_cfg=init_cfg)
        if num_convs == 0:
            assert self.in_channels == self.channels
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
        )
        for i in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )

        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = nn.Sequential(
                        nn.Conv2d(
                            self.in_channels + self.channels,
                            self.channels,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2),
                        nn.BatchNorm2d(self.channels),
                        nn.ReLU()
                    )
 
        if upsample_sacle is not None:
            self.upsample = Upsample(
                scale_factor=upsample_sacle,
                mode="bilinear",
                align_corners=False,
            )
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=3, padding=1)
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        self.dropout = None
                
    @property
    def with_upsample(self):
        return hasattr(self, "upsample") and self.upsample is not None

    def _forward_feature(self, x):
        feats = self.convs(x)
        # print(feats.shape)
        # print(x.shape)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.with_upsample:
            feat = self.upsample(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output



class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,**kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()

        self.depthwise_conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=in_channels,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU()
                )
        self.pointwise_conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
        )
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


@CHANGES.register_module()
class dwfcn(fcn):
    def __init__(self,**kwargs):
        super(dwfcn, self).__init__(**kwargs)
        self.convs[0] = DepthwiseSeparableConvModule(
            self.in_channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            )

        for i in range(1, self.num_convs):
            self.convs[i] = DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
                )

        if self.concat_input:
            self.conv_cat = DepthwiseSeparableConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2)


@CHANGES.register_module()
class ChangeSeg(BaseModule):
    def __init__(self,
    change,
    fcn,
    init_cfg= dict(type='Normal', std=0.01),**kwargs):
        super().__init__(init_cfg=init_cfg)

        self.change = CHANGES.build(change)
        self.fcn = CHANGES.build(fcn)

    def forward(self, input1,input2):
        diff = self.change(input1,input2)
        out = self.fcn(diff)
        return out


@BACKBONES.register_module()
class PDACN(BaseModule):
    def __init__(self, 
                backbone,
                seghead,
                changehead,
                aux_head=None,
                pretrained=None,
                init_cfg=None,
                ):
        super().__init__(init_cfg=init_cfg)
        if pretrained is not None:
            # assert (
            #     backbone.get("pretrained") is None
            # ), "both backbone and segmentor set pretrained weight"
            # backbone["pretrained"] = pretrained 
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        BACKBONES,SEGHEADS,CHANGES
        self.backbone = BACKBONES.build(backbone)
        self.seghead = SEGHEADS.build(seghead)
        self.changehead = CHANGES.build(changehead)
        if aux_head is not None:
            self.aux_head = CHANGES.build(aux_head)
        else:
            self.aux_head = None

    

    def forward(self, input1,input2):
        feats1 = self.backbone(input1)
        feats1 = self.seghead(feats1)

        feats2 = self.backbone(input2)
        feats2 = self.seghead(feats2)

        change_seg=self.changehead(feats1,feats2)

        return (change_seg,)



#--------------------------


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class MixVisionTransformerV0(BaseModule):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            pretrained=None,
            init_cfg=None,
            strides=[4, 2, 2, 2],
    ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=strides[0],
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=strides[1],
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=strides[2],
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=strides[3],
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformerV0, self).init_weights()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


@BACKBONES.register_module()
class mit_b0(MixVisionTransformerV0):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs,
        )


# @BACKBONES.register_module()
class mit_b1(MixVisionTransformerV0):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs,
        )



class mit_b2(MixVisionTransformerV0):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs,
        )



class mit_b3(MixVisionTransformerV0):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs,
        )



class mit_b4(MixVisionTransformerV0):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs,
        )



class mit_b5(MixVisionTransformerV0):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs,
        )


