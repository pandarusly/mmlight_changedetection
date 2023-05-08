import os
from PIL import Image
from mmcv.runner import Hook,HOOKS
import torch  
from mmseg.core import DistEvalHook, EvalHook
import numpy as np
import mmcv
import os.path as osp

from mmcd.datasets.pipelines.loading import MultiImgLoadImageFromFile
# 我在训练的时候，添加了'TensorboardLoggerHook'。我想在验证结束的时候，给Tensorboard添加部分图片和模型预测的结果。应该怎么实现？

# ?MMSegWandbHook


def single_gpu_test_for_seg_pred(model,data_loader,eval_image_indexs,target_layers=None):

    # # clean gpu memory when starting a new evaluation.
    # torch.cuda.empty_cache()
    
    model.eval() 
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, input_tensor in zip(loader_indices, data_loader): 
        if batch_indices[0] in eval_image_indexs: 
            with torch.no_grad():                 
                result = model(return_loss=False, **input_tensor)
                 
                results.extend(result)
        else: 
            results.extend(['result'])

        prog_bar.update() 

    return results

    
@HOOKS.register_module()
class CustomEvalHookV2(Hook):
    def __init__(self, iterval=10,num_eval_images=4,**kwargs):
        super().__init__(**kwargs)
        
        self.iterval = iterval
        self.num_eval_images = num_eval_images
        self.log_evaluation = (num_eval_images > 0) 

        self.eval_hook: EvalHook = None
        self.test_fn : single_gpu_test_for_seg_pred =None

    
    def before_run(self, runner):
        super().before_run(runner)
        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:
            if isinstance(hook, EvalHook):
                from mmseg.apis import single_gpu_test
                self.eval_hook = hook
                self.test_fn = single_gpu_test
            if isinstance(hook, DistEvalHook):
                from mmseg.apis import multi_gpu_test
                self.eval_hook = hook
                self.test_fn = multi_gpu_test

        self.val_dataset = self.eval_hook.dataloader.dataset
        # Determine the number of samples to be logged.
        if self.num_eval_images > len(self.val_dataset):
            self.num_eval_images = len(self.val_dataset)

        # Create directories
        self.image1_dir = os.path.join(runner.work_dir, self.val_dataset.sub_dir_1)
        self.image2_dir = os.path.join(runner.work_dir, self.val_dataset.sub_dir_2)
        self.label_dir = os.path.join(runner.work_dir, 'label')
        self.pred_dir = os.path.join(runner.work_dir, 'pred')
        for dir_path in [self.image1_dir, self.image2_dir, self.label_dir, self.pred_dir]:
            mmcv.mkdir_or_exist(dir_path)

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]


    def after_train_iter(self, runner):

        if self.every_n_iters(runner,self.iterval):
            # Currently the results of eval_hook is not reused by wandb, so
            # wandb will run evaluation again internally. We will consider
            # refactoring this function afterwards
            results = self.test_fn(runner.model, self.eval_hook.dataloader)


            for t in self.val_dataset.pipeline.transforms:
                if isinstance(t, MultiImgLoadImageFromFile):
                    img_loader = t  

            for idx in self.eval_image_indexs :
                img_info = self.val_dataset.img_infos[idx]
                image_name = img_info['filename']

                # Get image and convert from BGR to RGB
                img_meta = img_loader(dict(img_info=img_info, img_prefix=[osp.join(self.val_dataset.img_dir, self.val_dataset.sub_dir_1), \
                osp.join(self.val_dataset.img_dir, self.val_dataset.sub_dir_2)]))
                
                image1 =img_meta['img'][0]
                image2 = img_meta['img'][1] 

                # Get segmentation mask
                seg_mask = self.val_dataset.get_gt_seg_map_by_idx(idx)
                seg_pred = np.float32(results[idx])

                #  颜色表
                palette = np.zeros((len(self.val_dataset.PALETTE), 3), dtype=np.uint8)
                for label_id, color in enumerate(self.val_dataset.PALETTE):
                    palette[label_id] = color 
                seg_mask = Image.fromarray(seg_mask.astype(np.uint8)).convert('P')
                seg_pred = Image.fromarray(seg_pred.astype(np.uint8)).convert('P')

                seg_mask.putpalette(palette) 
                seg_pred.putpalette(palette) 

                # Save images and segmentation mask
                image1_path = os.path.join(self.image1_dir, image_name)
                image2_path = os.path.join(self.image2_dir, image_name)
                label_path = os.path.join(self.label_dir, image_name.replace("jpg","png"))
                if not os.path.exists(image1_path):
                    mmcv.imwrite(image1, image1_path)
                if not os.path.exists(image2_path):
                    mmcv.imwrite(image2, image2_path)
                if not os.path.exists(label_path): 
                    seg_mask.save(label_path)
 
                pred_path = os.path.join(self.pred_dir, '{}-step-{}.png'.format(image_name,runner.iter + 1))
                # mmcv.imwrite(seg_pred, pred_path) 
                seg_pred.save(pred_path)