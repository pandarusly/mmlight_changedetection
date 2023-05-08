import collections.abc
from abc import ABCMeta, abstractmethod, ABC
from itertools import repeat
from typing import Optional, Tuple, List

import einops
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import create_act_layer, create_attn
from torch import nn, einsum, Tensor

# from ..backbones import MAKEResNet
# from ..backbones.layers import resize, Upsample
from mmseg.models.builder import BACKBONES
from mmseg.ops import resize, Upsample



# -----------FPN

class ConvModule(nn.Module):
    def __init__(self, filters, kernel_size: int = 3, norm_cfg=None, conv_cfg=None,
                 act_cfg: str = "relu", inplace=True, **kwargs):
        super().__init__()
        layers = []
        for i in range(1, len(filters)):
            layers.extend([
                nn.Conv2d(filters[i - 1], filters[i], kernel_size, **kwargs),
                nn.BatchNorm2d(filters[i]) if norm_cfg else nn.Identity(),
                create_act_layer(
                    act_cfg, inplace=inplace) if act_cfg else nn.Identity()
            ])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.model(x)
        return x


class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                [in_channels[i],
                 out_channels],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                [out_channels,
                 out_channels],
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    [in_channels,
                     out_channels],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs


# -------------vvitConvModule


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding # 显著增加显存
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans

        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans,
                              embed_dim,
                              kernel_size=(1,) + patch_size,
                              stride=(1,) + patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """[summary]

        Args:
        x [b  c h w ]

        Returns:
            [type]: [b t (h w) c ] t=2
        """
        # padding
        # x = rearrange(x,"b t c h w -> b c t h w")
        # x = torch.stack([y1, y2], dim=2)
        _, _, T, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C T Wh Ww

        if self.norm is not None:
            x = rearrange(x, "b c t h w -> (b t) c h w")

            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)

            return rearrange(x, "(b t) (h w) c -> b t h w c ", c=self.embed_dim, t=T, h=Wh, w=Ww)

        # return rearrange(x, "b c t h w -> b  t (h w) c")
        return x.permute(0, 2, 3, 4, 1).contiguous()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Sattention(nn.Module):
    def __init__(self, dim, h, w, heads=8, dim_head=64, dropout=0., sr_ratio=1):
        super().__init__()

        self.h = h
        self.w = w

        project_out = not (heads == 1 and dim_head == dim)
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.sr_ratio = sr_ratio
        # 实现上这里等价于一个卷积层
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

    def forward(self, x):
        """
        x: (b, n, d)
        """
        b, n, d, h = *x.shape, self.heads

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        if self.sr_ratio > 1:

            x_ = rearrange(x[:, 1:], 'b n d -> b d n',
                           b=b).reshape(b, d, self.h, self.w)

            x_ = rearrange(self.sr(x_).reshape(
                b, d, -1), 'b d n ->b n d')

            # print(x[:, 0].shape)
            x_ = torch.cat([x[:, 0].unsqueeze(1), x_], dim=1)

            kv = self.to_kv(x_).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class STransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, h, w, mlp_dim, dropout=0., sr_ratio=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Sattention(dim, heads=heads,
                                        dim_head=dim_head, h=h, w=w, sr_ratio=sr_ratio, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x

            x = ff(x) + x

        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VVITv3(nn.Module):
    def __init__(self, image_size, dim, patch_size=1, num_frames=2, depth=2, heads=3, in_channels=3, dim_head=64,
                 sr_ratio=2,
                 dropout=0., emb_dropout=0.1, scale_dim=4, pos_norm=nn.LayerNorm):
        """[RSAvit模塊]

        Args:
            x ([b c t h w]): [description]

        Returns:
            [type]: [b c t h w]
        """
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        self.resolution = image_size // patch_size
        self.emb_dim = dim
        self.to_patch_embedding = PatchEmbed(patch_size=patch_size, in_chans=in_channels, embed_dim=dim,
                                             norm_layer=pos_norm)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = STransformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=dim * scale_dim, sr_ratio=sr_ratio,
            dropout=dropout, h=self.resolution, w=self.resolution)

        # self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=dim * scale_dim, dropout=dropout)

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        # shortcut = x  # (b c t h w)
        shortcut = self.to_patch_embedding(x)  # b t h w c

        x = rearrange(shortcut, "b t  h w c-> b  t (h w) c")

        b, t, n, l = x.shape

        # print(b*t*n*l == l*2*self.resolution**2) True

        cls_space_tokens = einops.repeat(
            self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]

        x = self.dropout(x)

        # break

        x = rearrange(x, 'b t n l -> (b t) n l')

        x = self.space_transformer(x)  # b t n+1 c

        out = x[:, 1:].view(b, t, self.resolution,
                            self.resolution, l)  # b t h w c

        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)  # b t c

        x = self.temporal_transformer(x)  # b t c 此可以操作 decoder 改進

        # out = shortcut + shortcut * x.unsqueeze(2).unsqueeze(2)
        out = out + out * x.unsqueeze(2).unsqueeze(2)

        return out


# ---------Unet

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class ASPP3d(nn.ModuleList):

    def __init__(self, features, out_features=None, dilations=(1, 2, 4, 8), act_cfg=None, **kwargs):
        super(ASPP3d, self).__init__()

        self.out_features = out_features if out_features else features
        self.dilations = dilations

        self.append(
            nn.Sequential(
                nn.Conv3d(features, self.out_features,
                          kernel_size=(2, 1, 1),
                          padding=(0, 0, 0),
                          dilation=(1,) + to_2tuple(dilations[0]),
                          bias=False),
                nn.BatchNorm3d(self.out_features),
                nn.Identity() if act_cfg is None else create_act_layer(act_cfg)
            )
        )

        for dilation in dilations[1:]:
            self.append(
                nn.Sequential(
                    nn.Conv3d(features, self.out_features,
                              kernel_size=(2, 3, 3),
                              padding=(0,) + to_2tuple(dilation),
                              dilation=(1,) + to_2tuple(dilation),
                              bias=False),
                    nn.BatchNorm3d(self.out_features),
                    nn.Identity() if act_cfg is None else create_act_layer(act_cfg)
                )
            )

    def forward(self, x):
        """Forward function."""
        # fusion_feas = torch.stack([A, B], dim=2)

        h, w = x.size()[-2:]
        # print(x.shape)
        aspp_outs = []
        for idx, aspp_module in enumerate(self):
            aspp_outs.append(aspp_module(x).squeeze(2)) if idx > 0 else aspp_outs.append(
                resize(aspp_module(x).squeeze(2), size=(h, w)))

        return aspp_outs


class AsppFusion(nn.Module):
    def __init__(self, features, out_features=None, dilations=(1, 8, 12, 24), act_cfg="relu"):
        super().__init__()

        self.ppm = ASPP3d(features=features, out_features=out_features,
                          dilations=dilations, act_cfg=act_cfg)

        self.bottleneck = ConvModule(
            [len(dilations) * self.ppm.out_features, self.ppm.out_features],
            3,
            padding=1,
            act_cfg=act_cfg) if len(dilations) > 1 else nn.Identity()

    def forward(self, x, y):
        fusion_fea = torch.stack([x, y], dim=2)
        fusion_fea = torch.cat(self.ppm(fusion_fea), dim=1)
        # print(fusion_fea.shape)
        fusion_fea = self.bottleneck(fusion_fea)

        return fusion_fea


class Aspp3dFusion(nn.Module):
    def __init__(self, in_channels, dilations=(1, 8, 12, 24), act_cfg="relu"):
        assert isinstance(in_channels, (list, tuple))

        super().__init__()
        self.ppms = nn.ModuleList([AsppFusion(
            features=channels, dilations=dilations, act_cfg=act_cfg) for channels in in_channels])

    def forward(self, x1, x2):
        """[summary]
        将 （b c h w）x2 ->  b c h w

        Args:
            x1 ([type]): [（b c h w）]
            x2 ([type]): [（b c h w）]
            output :（b 2c  h w）
        """
        # assert isinstance(x1, (list, tuple))
        # assert isinstance(x2, (list, tuple))
        if not isinstance(x1, (list, tuple)):
            # print(x1.shape)
            # print(x1.shape)
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, y, ppm in zip(x1, x2, self.ppms):
            out.append(ppm(x, y))
        return out


class ResidualCls(nn.Module):
    def __init__(self, channels, numclass, act_layer="relu"):
        super().__init__()

        self.conv5x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=3 // 2, stride=1, bias=False),
            create_act_layer(act_layer),
            nn.BatchNorm2d(channels),

            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=3 // 2, stride=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1,
                      padding=1 // 2, stride=1, bias=False),
            create_act_layer(act_layer),
            nn.BatchNorm2d(channels),

        )

        self.fc = nn.Conv2d(channels, numclass, kernel_size=1,
                            padding=1 // 2, stride=1)

    def forward(self, x):
        conv5_x = self.conv5(x)
        conv5x3_x = self.conv5x3(x)

        out = self.fc(conv5_x + conv5x3_x)

        return out


# 代替减法


class FeatureAttention(nn.Module):
    """ Feature Attention Block """

    def __init__(self, channels: int, kernel_size: int, r: int = 8):
        super().__init__()
        self.model = nn.Sequential(
            Reduce("b c t h w -> b c () () ()", "mean"),
            nn.Conv3d(channels, channels // r, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv3d(channels // r, channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.model(x)


class TemporalAttention(nn.Module):
    """ Temporal Attention Block """

    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.model = nn.Sequential(
            Reduce("b c h w -> b c () ()", "mean"),
            nn.Conv2d(channels, channels // r, 1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(channels // r, channels, 1,
                      stride=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print( self.model(x).shape)
        return x * self.model(x)


class Conv3dFusion(nn.Module):
    def __init__(self, in_channels, kernel_size=3, norm_cfg=None):
        assert isinstance(in_channels, (list, tuple))

        super().__init__()

        self.temporal_attn = nn.ModuleList([nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(
                channels, affine=True) if norm_cfg else nn.Identity(),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(
                channels, affine=True) if norm_cfg else nn.Identity(),
            FeatureAttention(channels, kernel_size),
            Rearrange("b c t h w -> b (c t) h w"),
            TemporalAttention(channels * 2, channels * 2),
            nn.BatchNorm2d(
                channels * 2, affine=True) if norm_cfg else nn.Identity(),
            nn.ReLU()
        ) for channels in in_channels])

        # print(self.temporal_attn)

    def forward(self, x1, x2):
        """[summary]
        将 （b c h w）x2 ->  b c 2 h w 做 attention 后进行attention

        Args:
            x1 ([type]): [（b c h w）]
            x2 ([type]): [（b c h w）]
            output :（b 2c  h w）
        """
        # assert isinstance(x1, (list, tuple))
        # assert isinstance(x2, (list, tuple))
        if not isinstance(x1, (list, tuple)):
            x1 = [x1]
            x2 = [x2]

        out = []
        for x, fusion_model in zip(list(zip(x1, x2)), self.temporal_attn):
            # print(x[0].shape)
            out.append(fusion_model(torch.stack(x, dim=2)))
        return out


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """
    基础网络框架，实现最终预测,可以自己选择特征进行预测
    """

    def __init__(self, in_channels, channels, num_classes, fusion_form="abs_diff", in_index=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg="relu",
                 first_fusion: Optional[bool] = True,
                 input_transform=None, align_corners=False, dropout_ratio=0.1):

        super(BaseDecodeHead, self).__init__()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.fusion_form = fusion_form

        self.first_fusion = first_fusion
        self.FUSION_DIC = {"2to1_fusion": ["sum", "diff", "abs_diff", "aspp3d"],
                           "2to2_fusion": ["concat", "3dconcat"],
                           "not_siam": ["t1t2", "t2t1"]}
        ##########################################################
        #     针对变化检测孪生网络做的判断
        #     elif not self.first_fusion and self.fusion_form == "aspp3d":
        # if isinstance(self.in_channels, (list, tuple)):
        #     self.conv_3d_fusion = Aspp3dFusion(
        #         in_channels=self.in_channels)
        # else:
        #     self.conv_3d_fusion = Aspp3dFusion(
        #         in_channels=[self.in_channels])
        ##########################################################
        # 后融
        if not self.first_fusion:
            if self.fusion_form in self.FUSION_DIC["2to2_fusion"]:
                self.two_layer_seg = ResidualCls(
                    self.channels * 2, self.num_classes)
                if self.fusion_form == "3dconcat":
                    if isinstance(self.channels, (list, tuple)):
                        self.conv_3d_fusion = Conv3dFusion(
                            self.channels, norm_cfg=True)
                    else:
                        self.conv_3d_fusion = Conv3dFusion(
                            [self.channels], norm_cfg=True)
            else:
                print("yes")
                self.two_layer_seg = ResidualCls(
                    self.channels, self.num_classes)
                if isinstance(self.channels, (list, tuple)):
                    self.conv_3d_fusion = Aspp3dFusion(
                        in_channels=self.channels, dilations=(1,))
                else:
                    self.conv_3d_fusion = Aspp3dFusion(
                        in_channels=[self.channels], dilations=(1,))

        else:  # 先融

            self.two_layer_seg = ResidualCls(
                self.channels, self.num_classes)
            if self.fusion_form in self.FUSION_DIC["2to2_fusion"]:
                if isinstance(self.in_channels, (list, tuple)):
                    self.in_channels = [x * 2 for x in self.in_channels]
                else:
                    self.in_channels = self.in_channels * 2
                # print(self.in_channels)
                if self.fusion_form == "3dconcat":
                    if isinstance(self.in_channels, (list, tuple)):
                        self.conv_3d_fusion = Conv3dFusion(
                            [x // 2 for x in self.in_channels], norm_cfg=True)  #
                        # 为了方便理解，其实放到 self.in_channels = self.in_channels * 2 之前就可以不除上二
                    else:
                        self.conv_3d_fusion = Conv3dFusion(
                            [self.in_channels // 2], norm_cfg=True)
            else:
                # print("yes111")
                if isinstance(self.in_channels, (list, tuple)):
                    self.conv_3d_fusion = Aspp3dFusion(
                        in_channels=self.in_channels, dilations=(1,))
                else:
                    self.conv_3d_fusion = Aspp3dFusion(
                        in_channels=[self.in_channels], dilations=(1,))
                # print(self.in_channels)
        self.upscale4 = Upsample(scale_factor=4)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def fusion(self, x1, x2, fusion_form="concat"):
        """Specify the form of feature fusion"""
        if fusion_form == "concat":
            x = torch.cat([x1, x2], dim=1)
        elif fusion_form == "sum":
            x = x1 + x2
        elif fusion_form == "diff":
            x = x2 - x1
        elif fusion_form == "abs_diff":
            x = torch.abs(x1 - x2)
        else:
            raise ValueError(
                'the fusion form "{}" is not defined'.format(fusion_form))

        return x

    def aggregation_layer(self, fea1, fea2, fusion_from="concat"):

        if fea2 is not None:
            if fusion_from == "3dconcat":
                aggregate_fea = self.conv_3d_fusion(fea1, fea1)
                return aggregate_fea

            if fusion_from == "aspp3d":
                aggregate_fea = self.conv_3d_fusion(fea1, fea2)
                return aggregate_fea

            else:
                if isinstance(fea1, (list, tuple)):
                    aggregate_fea = [self.fusion(fea1[idx], fea2[idx], fusion_from)
                                     for idx in range(len(fea1))]
                    return aggregate_fea
                else:
                    aggregate_fea = [self.fusion(fea1, fea2, fusion_from)]
                    return aggregate_fea

        else:
            if isinstance(fea1, (list, tuple)):
                return fea1
            else:
                return [fea1]


class SimpleHead(BaseDecodeHead, ABC):
    """
    简单的CNN卷积预测模块
    1\ 要用  multiple_select  需要改写 self.conv_pred 的逻辑
    config_decoder = dict(in_channels=in_channels[:2], in_index=[0, 1], channels=32, num_classes=2,
                      fusion_form="diff",
                      conv_cfg=None,
                      norm_cfg=True,
                      act_cfg="relu",
                      first_fusion=True,
                      input_transform="resize_concat", align_corners=False, dropout_ratio=0)
    """

    def __init__(self, in_index=[0, 1, 2, 3], input_transform: str = "resize_concat", **kwargs):
        # assert input_transform in ["resize_concat", None]
        super().__init__(input_transform=input_transform, in_index=in_index, **kwargs)
        if input_transform in ["resize_concat", None]:
            self.conv_pred = ConvModule([self.in_channels, self.channels], kernel_size=3, padding=1, act_cfg="relu",
                                        norm_cfg=True)
        self.upsamplex2 = Upsample(scale_factor=2)
        self.upsamplex4 = Upsample(
            scale_factor=4, mode="bilinear", align_corners=False)
        # self.first_attn = Residual(create_attn("se",self.in_channels))

    def forward(self, x1, x2=None):
        im_size = x1[0].size()[2:]
        if x2 is not None:
            inputs_ = [self._transform_inputs(x1), self._transform_inputs(x2)]
        else:
            assert self.first_fusion == True
            assert self.fusion_form in self.FUSION_DIC["not_siam"]
            inputs_ = [self._transform_inputs(x1), x2]

        if self.first_fusion:
            features = self.aggregation_layer(
                inputs_[0], inputs_[1], fusion_from=self.fusion_form)  # list

            features = self.forward_single(features[0])

        else:
            x1 = self.forward_single(inputs_[0])
            x2 = self.forward_single(inputs_[1])

            features = self.aggregation_layer(
                [x1], [x2], fusion_from=self.fusion_form)[0]  # 返回 list 后容需要直接输出

        features = resize(features, size=im_size,
                          mode='bilinear', align_corners=self.align_corners)
        features = self.upscale4(features)
        features = self.two_layer_seg(features)

        return features

    def forward_single(self, inputs: Tensor):
        x = self.conv_pred(inputs)  # 512 ->32 如果是 multiple_select 则是实现高级应用
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            attention_type="se",
            use_batchnorm=None,
    ):
        super().__init__()
        self.conv1 = ConvModule([in_channels + skip_channels, out_channels], kernel_size=3, padding=1,
                                norm_cfg=use_batchnorm)
        self.attention1 = create_attn(
            attention_type, in_channels + skip_channels)
        self.conv2 = ConvModule(
            [out_channels, out_channels], kernel_size=3, padding=1, norm_cfg=use_batchnorm)
        self.attention2 = create_attn(attention_type, out_channels)
        self.upscale_x2 = Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, skip=None):
        if skip is not None:
            x = resize(x, skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            # x = self.attention1(x) #None

        x = self.conv1(x)

        x = self.conv2(x)
        # x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = ConvModule(
            [in_channels,
             out_channels],
            kernel_size=3,
            padding=1,
            norm_cfg=True,
        )
        conv2 = ConvModule(
            [out_channels,
             out_channels],
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


"""只能用 multiple_select 
    config_decoder = dict(in_channels=in_channels[:2], in_index=[0, 1], channels=32, num_classes=2,
                          fusion_form="diff",
                          conv_cfg=None,
                          norm_cfg=True,
                          act_cfg="relu",
                          first_fusion=True,
                          input_transform="multiple_select", align_corners=False, dropout_ratio=0)
"""


class UnetHeadTransMuti(SimpleHead):
    def __init__(
            self,
            uout_channels: Optional[List] = None,
            use_batchnorm=True,
            attention_type=False,
            center=False,
            enchance=None,
            dilations=(1,),
            **kwargs
    ):

        super().__init__(**kwargs)

        if enchance is None:
            enchance = dict(
                patch_size=4, depth=2, in_channels=32,
                dim=128, dropout=0., heads=3, dim_head=64, image_size=32, sr_ratio=1
            )
        self.enchace = enchance

        if attention_type:
            print('UnetHead attention_type is {}'.format(attention_type))

        if self.enchace:
            self.fa = VVITv3(**enchance)
            self.in_channels[-1] = self.fa.emb_dim
            if self.first_fusion:
                self.conv_3d_fusion = Aspp3dFusion(
                    in_channels=self.in_channels, dilations=dilations)

        if uout_channels is None:
            uout_channels = self.in_channels[::-1]
            uout_channels[-1] = self.channels
            # print(uout_channels)

        if len(self.in_channels) != len(uout_channels):
            raise ValueError(
                "Model depth is {}, but you provide `uout_channels` for {} blocks.".format(
                    len(self.in_channels), len(uout_channels)
                )
            )

        encoder_channels = self.in_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(uout_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = uout_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        # print(kwargs)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        # ------------------------------------添加多loss约束
        # print(uout_channels)
        # print(self.num_classes)

        self.predicts = nn.ModuleList([ResidualCls(uout_channels[0], self.num_classes), ResidualCls(
            uout_channels[1], self.num_classes), ResidualCls(uout_channels[2], self.num_classes),
                                       ResidualCls(uout_channels[3], self.num_classes)])
        self.predict5 = nn.Sequential(self.upscale4, ResidualCls(
            sum(uout_channels), self.num_classes))
        self.two_layer_seg = nn.Identity()

    def forward(self, x1, x2):
        if x2 is not None:
            inputs_ = [self._transform_inputs(x1), self._transform_inputs(x2)]
        else:
            assert self.first_fusion == True
            assert self.fusion_form in self.FUSION_DIC["not_siam"]
            inputs_ = [self._transform_inputs(x1), x2]

        # b  c t h w  -> b t h w c
        if self.enchace:
            fuse_head = torch.stack([inputs_[0][-1], inputs_[1][-1]], dim=2)
            # print(fuse_head.shape)
            fuse_head = self.fa(fuse_head).permute(
                1, 0, 4, 2, 3).contiguous()  # t b c h w

            inputs_[0][-1] = fuse_head[0]
            inputs_[1][-1] = fuse_head[1]

        if self.first_fusion:
            features = self.aggregation_layer(
                inputs_[0], inputs_[1], fusion_from=self.fusion_form)
            output = self.forward_single(features)

        else:
            outputs = []
            for inputs in inputs_:
                outputs.append(self.forward_single(inputs))

            output = self.aggregation_layer(
                [outputs[0]], [outputs[1]], fusion_from=self.fusion_form)[0]

        predict_out = []
        for pred, uout in zip(self.predicts[:-1], output[:-1]):
            predict_out.append(pred(uout))

        predict_out.append(self.predicts[-1](self.upscale4(output[-1])))
        # ---------------------------------------cat pred
        # upsampled_output = [
        #     resize(
        #         input=x,
        #         size=output[-1].shape[2:],
        #         mode='bilinear',
        #         align_corners=False) for x in output
        # ]
        # upsampled_output = torch.cat(upsampled_output, dim=1)

        # predict_out.append(self.predict5(upsampled_output))

        return predict_out

    def forward_single(self, inputs):
        inputs = inputs[::-1]

        head = inputs[0]
        skips = inputs[1:]

        output = self.center(head)
        uout = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            output = decoder_block(output, skip)
            uout.append(output)

        return uout


class UnetHeadTransSingle(SimpleHead):
    def __init__(
            self,
            uout_channels: Optional[List] = None,
            use_batchnorm=True,
            attention_type=False,
            center=False,
            enchance=None,
            dilations=(1,),
            **kwargs
    ):

        super().__init__(**kwargs)

        if enchance is None:
            enchance = dict(
                patch_size=4, depth=2, in_channels=32,
                dim=128, dropout=0., heads=3, dim_head=64, image_size=32, sr_ratio=1
            )
        self.enchace = enchance

        if attention_type:
            print('UnetHead attention_type is {}'.format(attention_type))

        if self.enchace:
            self.fa = VVITv3(**enchance)
            self.in_channels[-1] = self.fa.emb_dim
            if self.first_fusion:
                self.conv_3d_fusion = Aspp3dFusion(
                    in_channels=self.in_channels, dilations=dilations)

        if uout_channels is None:
            uout_channels = self.in_channels[::-1]
            uout_channels[-1] = self.channels
            # print(uout_channels)

        if len(self.in_channels) != len(uout_channels):
            raise ValueError(
                "Model depth is {}, but you provide `uout_channels` for {} blocks.".format(
                    len(self.in_channels), len(uout_channels)
                )
            )

        encoder_channels = self.in_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(uout_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = uout_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        # print(kwargs)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x1, x2):
        if x2 is not None:
            inputs_ = [self._transform_inputs(x1), self._transform_inputs(x2)]
        else:
            assert self.first_fusion == True
            assert self.fusion_form in self.FUSION_DIC["not_siam"]
            inputs_ = [self._transform_inputs(x1), x2]

        # b  c t h w  -> b t h w c
        if self.enchace:
            fuse_head = torch.stack([inputs_[0][-1], inputs_[1][-1]], dim=2)
            # print(fuse_head.shape)
            fuse_head = self.fa(fuse_head).permute(
                1, 0, 4, 2, 3).contiguous()  # t b c h w

            inputs_[0][-1] = fuse_head[0]
            inputs_[1][-1] = fuse_head[1]

        if self.first_fusion:
            features = self.aggregation_layer(
                inputs_[0], inputs_[1], fusion_from=self.fusion_form)
            output = self.forward_single(features)

        else:
            outputs = []
            for inputs in inputs_:
                outputs.append(self.forward_single(inputs))

            output = self.aggregation_layer(
                [outputs[0]], [outputs[1]], fusion_from=self.fusion_form)[0]

        output = self.upscale4(output)

        output = self.two_layer_seg(output)

        return output

    def forward_single(self, inputs):
        inputs = inputs[::-1]

        head = inputs[0]
        skips = inputs[1:]

        output = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            output = decoder_block(output, skip)

        return output


# ---- decoder
from mmcv.runner import BaseModule
@BACKBONES.register_module()
class UVASPMuti(BaseModule):
    def __init__(self,
                 backbone,
                 feature_info,
                 neck=None,
                 encoder_set=None,
                 decoder_set=None, 
                init_cfg=None,
                pretrained=None,
                **kwargs
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

        if decoder_set is None:
            decoder_set = {'num_classes': 2, 'act_cfg': 'relu', 'channels': 32, 'in_index': [0, 1, 2, 3],
                           'input_transform': 'multiple_select', 'fusion_form': 'aspp3d', 'first_fusion': True,
                           'dilations': [1, 4, 12], 'attention_type': False,
                           'enchance': {'image_size': 32, 'in_channels': 64, 'patch_size': 1, 'depth': 2, 'sr_ratio': 1,
                                        'dim': 128, 'dropout': 0, 'emb_dropout': 0, 'heads': 4, 'dim_head': 256}}
        if encoder_set is None:
            encoder_set = {'output_stride': 8, 'in_chans': 3}
        if neck is None:
            neck = {'with_neck': True, 'neck_out': 64}
        self.backbone = BACKBONES.build(backbone)
        # 创建 FPN增强--------------------------------
         
        self.with_neck = neck["with_neck"]
        self.neck = FPN(feature_info, neck["neck_out"],
                        len(feature_info))
        if self.with_neck:
           feature_info = [
                neck["neck_out"] for x in feature_info]

        # --------------------------------

        fea1_nums = self.filter_head(
           feature_info, decoder_set["in_index"])  # [64, 128, 256, 512]

        self.decoder = UnetHeadTransMuti(in_channels=fea1_nums, **decoder_set)

    @staticmethod
    def filter_head(fea1_nums, head_index=-1):
        if isinstance(head_index, list):
            fea1_nums_ = [fea1_nums[x] for x in head_index]
            return fea1_nums_
        else:
            return fea1_nums[head_index]

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, x1, x2):

        fea1 = self.extract_feat(x1)
        fea2 = self.extract_feat(x2)

        return self.decoder(fea1, fea2)
