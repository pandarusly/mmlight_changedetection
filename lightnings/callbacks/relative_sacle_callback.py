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


class RelativeSacleCall(Callback):

    def __init__(self,
                 every_n_epoches=2,
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

            
            imgs= batch['img'][0]
            img_metas  =  batch['img_metas'][0]
            show_epoch = trainer.current_epoch % self.evevery_n_epoches
            if show_epoch == 0 and batch_idx in self.batch_ids:
                
                with torch.no_grad():
                    x, y = torch.chunk(imgs, chunks=2, dim=1)
                    inputs = pl_module.model.extract_feat(imgs)
                    ChangeFCNHead = pl_module.model.decode_head[0]
                    x1, x2 = ChangeFCNHead._transform_inputs(inputs)
                    x1 = ChangeFCNHead.pre_process(x1)
                    x2 = ChangeFCNHead.pre_process(x2)

                    # u = ChangeFCNHead.difference.forward_step(x1, x2)
                    # v = ChangeFCNHead.difference.forward_step(x2, x1)
                    Across1 = ChangeFCNHead.difference.scale_12(
                        ChangeFCNHead.difference.cross_attention12(x1, x2))
                    Across2 = ChangeFCNHead.difference.scale_21(
                        ChangeFCNHead.difference.cross_attention21(x2, x1))

                    u = x1 + Across1
                    v = x2 + Across2

                    attn_diff = torch.abs(u - v)

                x1 = self.saliency_map(align_feat(
                    x1, imgs)) * 255
                x2 = self.saliency_map(align_feat(
                    x2, imgs)) * 255
                show3 = self.saliency_map(align_feat(
                    u, imgs)) * 255
                show4 = self.saliency_map(align_feat(
                    v, imgs)) * 255
                show5 = self.saliency_map(align_feat(
                    attn_diff, imgs)) * 255

                show_pic = [x1, x2, show3, show4, show5]
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
                plt.close("all")

# │         test.mIoU         │    0.9259999999999999     │
# │         test.mIoU         │    0.9386      │
# │         test.mIoU         │    0.9315000000000001     │
#  test.mIoU         │          0.9458           │
# test.mIoU         │    0.9437000000000001     │


class valid_pdaCall(RelativeSacleCall):
    def __init__(self, every_n_epoches=1, batch_ids=[1, 2, 3]) -> None:
        super().__init__(every_n_epoches, batch_ids)


    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
     
        
        def align_feat(feat, img): return resize(
                                input=feat, size=img.shape[-2:], mode='bilinear', align_corners=pl_module.model.align_corners)
        

        imgs= batch['img'][0]
        img_metas  =  batch['img_metas'][0]
        
        # if img_metas[0]['filename'] == 'test_18_1_3.png':
        if  batch_idx in self.batch_ids:
            print(img_metas[0]['filename'])
            with torch.no_grad():
                img1, img2 = torch.chunk(imgs, chunks=2, dim=1)
                inputs = pl_module.model.extract_feat(imgs)
                ChangeFCNHead = pl_module.model.decode_head
                x1, x2 = ChangeFCNHead._transform_inputs(inputs)
                x1 = ChangeFCNHead.pre_process(x1)
                x2 = ChangeFCNHead.pre_process(x2)
                u = x1.clone()
                v = x2.clone()
                attn = torch.stack([x1, x2], dim=2)  # b c t h w
                attn = ChangeFCNHead.difference.conv0(attn).squeeze(2)  # b c h w
                attn = ChangeFCNHead.difference.point_wise(attn)
                u = u * attn
                u = ChangeFCNHead.difference.conv1(u)
                v = v * attn
                v = ChangeFCNHead.difference.conv1(v)

                attn_diff = torch.abs(u - v)

            x1 = self.saliency_map(align_feat(
                x1, imgs)) * 255
            x2 = self.saliency_map(align_feat(
                x2, imgs)) * 255
            show3 = self.saliency_map(align_feat(
                u, imgs)) * 255
            show4 = self.saliency_map(align_feat(
                v, imgs)) * 255 
            show5 = self.saliency_map(align_feat(
                attn_diff, imgs)) * 255

            show_pic = [x1, x2, show3, show4, show5]
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
            plt.close("all")

class test_RelativeSacleCall(RelativeSacleCall):
    def __init__(self, every_n_epoches=2, batch_ids=[1, 2, 3],test_name=[]) -> None:
        super().__init__(every_n_epoches, batch_ids)

        self.test_name=test_name
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        def align_feat(feat, img): return resize(
                        input=feat, size=img.shape[-2:], mode='bilinear', align_corners=pl_module.model.align_corners)
 

        imgs= batch['img'][0]
        img_metas  =  batch['img_metas'][0]
        
        # if img_metas[0]['filename'] == 'test_18_1_3.png':
        if  batch_idx in self.batch_ids:
            print(img_metas[0]['filename'])
            with torch.no_grad():
                x, y = torch.chunk(imgs, chunks=2, dim=1)
                inputs = pl_module.model.extract_feat(imgs)
                ChangeFCNHead = pl_module.model.decode_head
                x1, x2 = ChangeFCNHead._transform_inputs(inputs)
                x1 = ChangeFCNHead.pre_process(x1)
                x2 = ChangeFCNHead.pre_process(x2)
 
                Across1 = ChangeFCNHead.difference.scale_12(
                    ChangeFCNHead.difference.cross_attention12(x1, x2))
                Across2 = ChangeFCNHead.difference.scale_21(
                    ChangeFCNHead.difference.cross_attention21(x2, x1))

                u = x1 + Across1
                v = x2 + Across2

                attn_diff = torch.abs(u - v)

            x1 = self.saliency_map(align_feat(
                x1, imgs)) * 255
            x2 = self.saliency_map(align_feat(
                x2, imgs)) * 255
            show3 = self.saliency_map(align_feat(
                u, imgs)) * 255
            show4 = self.saliency_map(align_feat(
                v, imgs)) * 255 
            show5 = self.saliency_map(align_feat(
                attn_diff, imgs)) * 255

            show_pic = [x1, x2, show3, show4, show5]
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
            plt.close("all")
            