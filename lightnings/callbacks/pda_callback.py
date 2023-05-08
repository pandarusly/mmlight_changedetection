from typing import Any, Optional

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid

from mmseg.ops import resize


class CAMCall(Callback):

    def __init__(self,
                 every_n_epoches=30,
                 batch_ids=[1, 2, 3]
                 ) -> None:

        super().__init__()
        self.evevery_n_epoches = every_n_epoches
        self.batch_ids = batch_ids
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def saliency_map(self, x, mode="mean"):
        # (b c h w) -> ( b h w )
        xx = torch.maximum(x, torch.tensor([0]).to(x))
        xx = torch.mean(xx, dim=1) if mode == "mean" else torch.sum(xx, dim=1)
        xx = torch.div(xx, torch.max(xx)) 
        return xx

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:

        if self.ready:
            def align_feat(feat, img): return resize(
                input=feat, size=img.shape[-2:], mode='bilinear', align_corners=pl_module.model.align_corners)

            show_epoch = trainer.current_epoch % self.evevery_n_epoches
            if show_epoch == 0 and batch_idx in self.batch_ids:

                imgs, img_metas = batch['img'][0], batch
                ['img_metas'][0]

                x, y = torch.chunk(imgs, chunks=2, dim=1)
                x1 = pl_module.model.extract_single_time_feat(x)
                x2 = pl_module.model.extract_single_time_feat(y)
                x1, x2 = list(x1)[0], list(x2)[0]

                fusion = pl_module.model.difference.fusion[0]
                u = x1.clone()
                v = x2.clone()
                attn = torch.stack([x1, x2], dim=2)  # b c t h w
                attn = fusion.conv0(attn).squeeze(2)  # b c h w
                attn = fusion.point_wise(attn)

                show1 = self.saliency_map(align_feat(
                    u, imgs))* 255
                show2 = self.saliency_map(align_feat(
                    v, imgs))* 255
                show6 = self.saliency_map(align_feat(
                    attn, imgs))* 255

                u = u * attn
                u = fusion.conv1(u)
                v = v * attn
                v = fusion.conv1(v)
                attn_diff = torch.abs(u - v)

 
                show3 = self.saliency_map(align_feat(
                    u, imgs))* 255
                show4 = self.saliency_map(align_feat(
                    v, imgs))* 255
                show5 = self.saliency_map(align_feat(
                    attn_diff, imgs))* 255

 
                show_pic = [show1, show2,show6, show3, show4, show5]
                plt.figure(figsize=(28, 15))  # 设置窗口大小
                from matplotlib import colors
                norm = colors.Normalize(0, 255)
                ncols = len(show_pic)
                fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
                all_ax = []
                for id, x_ in enumerate(show_pic):
                    x_ = x_.cpu().detach().numpy().astype(np.uint8)
                    axs[id].axis("off")
                    # im = axs[id].imshow(x_,norm=norm,cmap='rainbow')
                    im = axs[id].imshow(x_[0], norm=norm, cmap='jet')
                    all_ax.append(axs[id])
                # fig.colorbar(im,ax=all_ax,fraction=0.03,pad=0.05)
                fig.colorbar(im, ax=all_ax, fraction=0.03, pad=0.05)
                # plt.show()
                trainer.logger.experiment.add_figure(
                    "pda_batch_idx_%i" % batch_idx, fig, global_step=trainer.current_epoch)


class CAMCallV2(CAMCall):
    def __init__(self, every_n_epoches=30, batch_ids=[1, 2, 3]) -> None:
        super().__init__(every_n_epoches, batch_ids)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:


        if self.ready:
            def align_feat(feat, img): return resize(
                input=feat, size=img.shape[-2:], mode='bilinear', align_corners=pl_module.model.align_corners)

            show_epoch = trainer.current_epoch % self.evevery_n_epoches
            if show_epoch == 0 and batch_idx in self.batch_ids:

                imgs, img_metas = batch['img'][0], batch
                ['img_metas'][0]

                x, y = torch.chunk(imgs, chunks=2, dim=1)
                x1 = pl_module.model.extract_single_time_feat(x)
                x2 = pl_module.model.extract_single_time_feat(y)
                x1, x2 = list(x1)[0], list(x2)[0]




                fusion = pl_module.model.difference.fusion[0]
                u = x1.clone()
                v = x2.clone()
                attn = torch.stack([x1, x2], dim=2)  # b c t h w
                attn = fusion.conv0(attn).squeeze(2)  # b c h w
                
                attn_0 = fusion.conv0_1(attn)
                attn_0 = fusion.conv0_2(attn_0)

                attn_1 = fusion.conv1_1(attn)
                attn_1 = fusion.conv1_2(attn_1)

                attn_2 = fusion.conv2_1(attn)
                attn_2 = fusion.conv2_2(attn_2)
                attn = attn + attn_0 + attn_1 + attn_2

                attn = fusion.point_wise(attn)

                show1 = self.saliency_map(align_feat(
                    u, imgs))* 255
                show2 = self.saliency_map(align_feat(
                    v, imgs))* 255
                show6 = self.saliency_map(align_feat(
                    attn, imgs))* 255

                u = u * attn
                u = fusion.conv1(u)
                v = v * attn
                v = fusion.conv1(v)
                attn_diff = torch.abs(u - v)

 
                show3 = self.saliency_map(align_feat(
                    u, imgs))* 255
                show4 = self.saliency_map(align_feat(
                    v, imgs))* 255
                show5 = self.saliency_map(align_feat(
                    attn_diff, imgs))* 255

 
                show_pic = [show1, show2,show6, show3, show4, show5]
                plt.figure(figsize=(28, 15))  # 设置窗口大小
                from matplotlib import colors
                norm = colors.Normalize(0, 255)
                ncols = len(show_pic)
                fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
                all_ax = []
                for id, x_ in enumerate(show_pic):
                    x_ = x_.cpu().detach().numpy().astype(np.uint8)
                    axs[id].axis("off")
                    # im = axs[id].imshow(x_,norm=norm,cmap='rainbow')
                    im = axs[id].imshow(x_[0], norm=norm, cmap='jet')
                    all_ax.append(axs[id])
                # fig.colorbar(im,ax=all_ax,fraction=0.03,pad=0.05)
                fig.colorbar(im, ax=all_ax, fraction=0.03, pad=0.05)
                # plt.show()
                trainer.logger.experiment.add_figure(
                    "pdav2_batch_idx_%i" % batch_idx, fig, global_step=trainer.current_epoch)

