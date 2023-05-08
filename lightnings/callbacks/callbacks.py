
from pytorch_lightning.callbacks import ModelCheckpoint
import  pytorch_lightning as pl
from weakref import proxy
# pytorch_lightning 中如何自定义保存的权重参数，例如为每个模型参数去掉 'model'前缀。怎么实现？

# 我现在有两个类 A,B，我现在想让 B 所有的属性全都放到A中去。具体的实现步骤是，A初始化的过程中，先将B实例化，然后把B的所有属性都A。 怎么用python实现？

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


import torch


a = torch.load(r'logs\veshi\runs\2023-03-04_20-02-38\checkpoints\epoch_003_mIoU0.741.ckpt')

a.keys()

print(a)
