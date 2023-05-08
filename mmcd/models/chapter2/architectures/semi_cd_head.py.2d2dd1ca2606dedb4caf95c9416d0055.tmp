from mmcd.models.chapter1.change_decoders.tia_head import TIAHead
from mmseg.models.builder import HEADS
from torch import nn

import torch
import torch.nn.functional as F
from torch import nn 
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform



@HEADS.register_module()
class SemiCDTIAHead(TIAHead):
    def __init__(self,upscale = 4 , attn=None, difference_cfg=..., fuse_cfg=..., **kwargs):
        super().__init__(attn, difference_cfg, fuse_cfg, **kwargs)

        
        self.main_decoder   = MainDecoder(upscale, self.channels, num_classes=self.out_channels)

        

        conf = {
        "supervised": False,
        "semi": True,
        "supervised_w": 1,

        "sup_loss": "CE",
        "un_loss": "MSE",

        "softmax_temp": 1,
        "aux_constraint": False,
        "aux_constraint_w": 1,
        "confidence_masking": False,
        "confidence_th": 0.5,

        "drop": 5,
        "drop_rate": 0.5,
        "spatial": True,
    
        "cutout": 5,
        "erase": 0.4,
    
        "vat": 2,
        "xi": 1e-6,
        "eps": 2.0,

        "context_masking": 2,
        "object_masking": 2,
        "feature_drop": 5,

        "feature_noise": 5,
        "uniform_range": 0.3
        }
        # Use weak labels
        # self.use_weak_lables= use_weak_lables
        # self.weakly_loss_w  = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint     = conf['aux_constraint']
        self.aux_constraint_w   = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th      = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

         
        self.unsuper_loss = softmax_mse_loss 

        vat_decoder     = [VATDecoder(upscale, self.channels, self.out_channels, xi=conf['xi'],
            							eps=conf['eps']) for _ in range(conf['vat'])]
        drop_decoder    = [DropOutDecoder(upscale, self.channels, self.out_channels,
                                    drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
                                    for _ in range(conf['drop'])]
        cut_decoder     = [CutOutDecoder(upscale, self.channels, self.out_channels, erase=conf['erase'])
                                    for _ in range(conf['cutout'])]
        context_m_decoder = [ContextMaskingDecoder(upscale, self.channels, self.out_channels)
                                    for _ in range(conf['context_masking'])]
        object_masking  = [ObjectMaskingDecoder(upscale, self.channels, self.out_channels)
                                    for _ in range(conf['object_masking'])]
        feature_drop    = [FeatureDropDecoder(upscale, self.channels, self.out_channels)
                                    for _ in range(conf['feature_drop'])]
        feature_noise   = [FeatureNoiseDecoder(upscale, self.channels, self.out_channels,
                                    uniform_range=conf['uniform_range'])
                                    for _ in range(conf['feature_noise'])]

        self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                    *context_m_decoder, *object_masking, *feature_drop, *feature_noise])
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        f1 = self.fuse_conv(inputs1)
        f2 = self.fuse_conv(inputs2)

        if self.netA:
            f1, f2 = self.netA(f1, f2)
 
        out = self.calc_dist(f1, f2)

        out = self.main_decoder(out)

        return out

    def sup_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def unsup_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        f1 = self.fuse_conv(inputs1)
        f2 = self.fuse_conv(inputs2)

        if self.netA:
            f1, f2 = self.netA(f1, f2)

        # ----------------------------

        x_ul = self.calc_dist(f1, f2)
        output_ul = self.main_decoder(x_ul)

        # Get auxiliary predictions
        outputs_ul = [aux_decoder(x_ul, output_ul.detach(), pertub=True) for aux_decoder in self.aux_decoders]
        targets = F.softmax(output_ul.detach(), dim=1)

        # Compute unsupervised loss
        loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                        conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                        for u in outputs_ul])
        loss_unsup = (loss_unsup / len(outputs_ul))


        losses = {'loss_unsup': loss_unsup} 
        return losses

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        if gt_semantic_seg is not None:
            losses = self.sup_train(inputs, img_metas, gt_semantic_seg, train_cfg)
        else:
            losses = self.unsup_train(inputs, img_metas, None, train_cfg)

        return losses

         
 



# ------------------------ decoders



def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


# def upsample(in_channels, out_channels, upscale, kernel_size=3):
#     # A series of x 2 upsamling until we get to the upscale we want
#     layers = []
#     conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#     nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
#     layers.append(conv1x1)
#     for i in range(int(math.log(upscale, 2))):
#         layers.append(PixelShuffle(out_channels, scale=2))
#     return nn.Sequential(*layers)

def upsample(in_channels, out_channels, upscale=None, kernel_size=3):
    layers = []

    # Middle channels
    mid_channels = 32

    #First conv layer to reduce number of channels
    diff_conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False)
    nn.init.kaiming_normal_(diff_conv1x1.weight.data, nonlinearity='relu')
    layers.append(diff_conv1x1)

    #ReLU
    diff_relu = nn.ReLU()
    layers.append(diff_relu)

    #Upsampling to original size
    up      = nn.Upsample(scale_factor=upscale, mode='bilinear')
    layers.append(up)

    #Classification layer
    conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)

    return nn.Sequential(*layers)


class MainDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(MainDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x


class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _, pertub=True):
        if pertub:
            x = self.upsample(self.dropout(x))
        else:
            x = self.upsample(x)
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _, pertub=True):
        if pertub:
            x = self.feature_dropout(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _, pertub=True):
        if pertub:
            x = self.feature_based_noise(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x



def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _, pertub=True):
        if pertub:
            r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
            x = self.upsample(x + r_adv)
        else:
            x = self.upsample(x)
        return x



def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6)/10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try: # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except: # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w-min_w, max_h-min_h
            rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
            rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
            h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
            w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None, pertub=True):
        if pertub:
            maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.size(2), x.size(3)))
            x = x * maskcut
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x


def guided_masking(x, output, upscale, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
    
    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None, pertub=True):
        if pertub:
            x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          upscale=self.upscale, return_msk_context=True)
            x_masked_context = self.upsample(x_masked_context)
        else:
            x_masked_context = self.upsample(x)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None, pertub=True):
        if pertub:
            x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
            x_masked_obj = self.upsample(x_masked_obj)
        else:
            x_masked_obj = self.upsample(x)
        return x_masked_obj


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size() # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean') # take the mean over the batch_size