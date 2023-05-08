
from mmseg.models.decode_heads.decode_head import BaseDecodeHead 
from mmseg.models.builder import HEADS
import torch
import torch.nn as nn  
from mmcd.models.chapter1.builder import build_attention
from mmcv.runner import BaseModule


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x): 
        x = self.conv1x1(x) 
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) 
        x += residual
        x = self.relu(x)
        return x


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(PixelShuffleUpsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class ResUNetPixelShuffleNeck(BaseModule):
    def __init__(self, input_channels, mid_channels):
        super(ResUNetPixelShuffleNeck, self).__init__()
        assert len(input_channels) > 1, "Input channels list should have at least two elements."
        assert mid_channels > 0, "Mid channels should be a positive integer."

        self.layers = nn.ModuleList()
        self.num_layers = len(input_channels) - 1
        self.upsample_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                input_channel = input_channels[i]
            else:
                input_channel = mid_channels
            self.upsample_layers.append(PixelShuffleUpsample(input_channel,2))
            self.layers.append(
                nn.Sequential(
                nn.Conv2d(input_channel, mid_channels, kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(mid_channels)
                )
            )
            self.layers.append(ResidualBlock(input_channels[i + 1] + mid_channels, mid_channels))

        # self.final_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, resnet_outputs):
        resnet_outputs = list(reversed(resnet_outputs))
        x = resnet_outputs[0]
        outs = [resnet_outputs[0]]
        for i in range(self.num_layers):
 
            if x.shape[2:] != resnet_outputs[i + 1].shape[2:]:
                x = self.upsample_layers[i](x) 
 
            x = self.layers[2 * i](x)
            x = torch.cat([x, resnet_outputs[i + 1]], dim=1)
            x = self.layers[2 * i + 1](x)
            outs.append(x)
 
        return tuple(reversed(outs))

 

@HEADS.register_module()
class SiameseUnetDecoder(BaseDecodeHead):
              
    def __init__(
            self, 
            difference_cfg=[],
            upscale_factor=4,
            **kwargs):
        super().__init__(input_transform='multiple_select',**kwargs)
        assert isinstance(difference_cfg,list)

        in_channels = self.in_channels
        channels = self.channels

        self.siamese_unet = ResUNetPixelShuffleNeck(
            input_channels=list(reversed(in_channels)), 
            mid_channels=channels
        ) # --- [c1, c2, c3, c4] -> reverse -> [c, c, c, c4]
       
        self.change_unet = ResUNetPixelShuffleNeck(
            input_channels=[in_channels[-1] if i==0 else channels for i in range(len(in_channels))], 
            mid_channels=channels
        ) # ---  [c, c, c, c4] x2 -> reverse -> [c, c, c, c4]


        self.calc_dist = torch.nn.ModuleList(
            [
            build_attention(difference_) for difference_ in difference_cfg
            ]
        )

        self.upscale = PixelShuffleUpsample(channels,upscale_factor)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = [] 
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        # -----------------
        # --- [c1, c2, c3, c4] -> [c, c, c, c4]
        inputs1 = self.siamese_unet(inputs1)
        inputs2 = self.siamese_unet(inputs2)
        # -----------------

        cc = []
        for id, calc_dist in  enumerate(self.calc_dist):
            cc.append(
                calc_dist(inputs1[id],inputs2[id])
            )

        out = self.change_unet(cc)[0]

        out = self.upscale(out) 

        out = self.cls_seg(out)

        return out



@HEADS.register_module()
class UnetDecoder(BaseDecodeHead):
    def __init__(self,difference_cfg,
                 upscale_factor=4,
                 **kwargs):
        super().__init__(input_transform='multiple_select',**kwargs)

        in_channels = self.in_channels
        channels = self.channels
 
        self.change_unet = ResUNetPixelShuffleNeck(
            input_channels=list(reversed(in_channels)), 
            mid_channels=channels
        ) # --- [c1, c2, c3, c4] -> reverse -> [c, c, c, c4]

        assert isinstance(difference_cfg,list)
        self.calc_dist = torch.nn.ModuleList(
            [
            build_attention(difference_) for difference_ in difference_cfg
            ]
        )

        self.upscale = PixelShuffleUpsample(channels,upscale_factor)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = [] 
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        # -----------------
        # --- [c1, c2, c3, c4] -> [c, c, c, c4]
        cc = []
        for id, calc_dist in  enumerate(self.calc_dist):
            cc.append(
                calc_dist(inputs1[id],inputs2[id])
            )
        out = self.change_unet(cc)[0]

        out = self.upscale(out)

        out = self.cls_seg(out)

        return out


 
        
        

