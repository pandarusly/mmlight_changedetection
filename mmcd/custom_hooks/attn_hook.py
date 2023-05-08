import os
from mmcv.runner import Hook,HOOKS 
from mmseg.core import EvalHook
import numpy as np
import mmcv
import os.path as osp

from .cam_utils import single_gpu_test_for_feature

from mmcd.datasets.pipelines.loading import MultiImgLoadImageFromFile

from pytorch_grad_cam.utils.image import show_cam_on_image 
from PIL import Image

 
# ?MMSegWandbHook
@HOOKS.register_module()
class CustomEvalHook(Hook):
    def __init__(self, iterval=10,num_eval_images=4,layer_names = [''],**kwargs):
        super().__init__(**kwargs)
        
        self.iterval = iterval
        self.num_eval_images = num_eval_images
        self.log_evaluation = (num_eval_images > 0) 

        self.eval_hook: EvalHook = None
        self.test_fn : single_gpu_test_for_feature =None

 
    def before_run(self, runner):
        super().before_run(runner)
        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:
            if isinstance(hook, EvalHook): 
                self.eval_hook = hook

        self.test_fn = single_gpu_test_for_feature 


        self.val_dataset = self.eval_hook.dataloader.dataset
        # Determine the number of samples to be logged.
        if self.num_eval_images > len(self.val_dataset):
            self.num_eval_images = len(self.val_dataset)

        # Create directories
        self.image1_dir = os.path.join(runner.work_dir, self.val_dataset.sub_dir_1)
        self.image2_dir = os.path.join(runner.work_dir, self.val_dataset.sub_dir_2)
        self.label_dir = os.path.join(runner.work_dir, 'label')
        self.pred_dir = os.path.join(runner.work_dir, 'pred')
        self.cam_dir = os.path.join(runner.work_dir, 'cam')


        for dir_path in [self.image1_dir, self.image2_dir, self.label_dir, self.pred_dir, self.cam_dir]:
            mmcv.mkdir_or_exist(dir_path)

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]


    def after_train_iter(self, runner):
        # after_val_epoch

        if self.every_n_iters(runner,self.iterval):

            # Currently the results of eval_hook is not reused by wandb, so
            # wandb will run evaluation again internally. We will consider
            # refactoring this function afterwards 

            lka_layers = [runner.model.module.decode_head.fuse_conv, 
                          runner.model.module.decode_head.netA.proj2,
                         runner.model.module.decode_head.calc_dist,
                            ]
            

            grayscale_cams,results = self.test_fn(runner.model, \
                                                  self.eval_hook.dataloader,\
                                                  self.eval_image_indexs, \
                                                  target_layers=lka_layers
                                                )

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
                seg_mask = self.val_dataset.get_gt_seg_map_by_idx(idx) *255

                # Save images and segmentation mask
                image1_path = os.path.join(self.image1_dir, image_name)
                image2_path = os.path.join(self.image2_dir, image_name)
                label_path = os.path.join(self.label_dir, image_name)
                if not os.path.exists(image1_path):
                    mmcv.imwrite(image1, image1_path)
                    mmcv.imwrite(image2, image2_path)
                    mmcv.imwrite(seg_mask, label_path)
                
                if results is not None:
                    seg_pred = results[idx] *255
                    pred_path = os.path.join(self.pred_dir, '{}-step-{}.png'.format(image_name,runner.iter + 1))
                    mmcv.imwrite(seg_pred, pred_path) 

                grayscale_cam = grayscale_cams[idx]
                if len(grayscale_cam) >= 2:
                    for gc_idx, gc in enumerate(grayscale_cam):
                        img_size=gc.shape[-2:]
                        cam_image = show_cam_on_image(np.ones((img_size[0], img_size[1], 3)), gc[0, ...], use_rgb=True)
                        tmp_img = Image.fromarray(cam_image)
                        tmp_img.save( os.path.join(self.cam_dir,'{}-step-{}_cam{}.png'.format(image_name,runner.iter + 1,gc_idx+1)))

                        # mmcv.imwrite(cam_image, os.path.join(self.cam_dir,'{}-step-{}_cam{}.png'.format(image_name,runner.iter + 1,gc_idx+1))) 
                else:
                    img_size=grayscale_cam.shape[-2:]
                    cam_image = show_cam_on_image(np.ones((img_size[0], img_size[1], 3)), grayscale_cam[0][0, ...], use_rgb=True)

                    tmp_img = Image.fromarray(cam_image)
                    tmp_img.save(os.path.join(self.cam_dir,'{}-step-{}_camOne.png'.format(image_name,runner.iter + 1)))

                    # mmcv.imwrite(cam_image, os.path.join(self.cam_dir,'{}-step-{}_camOne.png'.format(image_name,runner.iter + 1))) 


            grayscale_cams,results,lka_layers = None,None,None