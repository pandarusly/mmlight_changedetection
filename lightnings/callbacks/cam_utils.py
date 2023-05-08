#  

import cv2
import mmcv
import torch


import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from typing import List, Tuple
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image,show_cam_on_image 
 
def logit_simple_test(self, img, img_meta, rescale=True):
    """Simple test with single image."""
    seg_logit = self.inference(img, img_meta, rescale)
    # unravel batch dim
    seg_pred = list(seg_logit)
    return seg_pred


class MMActivationsAndGradients(ActivationsAndGradients):
    def __init__(self, *args, **kwargs):
        super(MMActivationsAndGradients, self).__init__(*args, **kwargs)
    
    def __call__(self, x):
        
        self.gradients = []
        self.activations = []
        out = self.model(return_loss=False, **x)
        return out


class MMGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, use_siam_layer=True):
        super(
            MMGradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
            
        self.activations_and_grads = MMActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.use_siam_layer = use_siam_layer
        self.uses_gradients=False


    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor['img'][0].size(-1), input_tensor['img'][0].size(-2)
        return width, height
    
    def forward(self,
            input_tensor: dict,
            targets: List[torch.nn.Module],
            eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor['img'][0] = input_tensor['img'][0].cuda()

        input_tensor['img'][0] = torch.autograd.Variable(input_tensor['img'][0],
                                                requires_grad=False)
        outputs = self.activations_and_grads(input_tensor)
 
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        
 
        if self.use_siam_layer:
            return [self.aggregate_multi_layers([cam_per_layer[i]]) for i in range(len(cam_per_layer))], outputs 
        
        return (self.aggregate_multi_layers(cam_per_layer), outputs )

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
    
    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        # weights = self.get_cam_weights(input_tensor,
        #                                target_layer,
        #                                targets,
        #                                activations,
        #                                grads)
        # weighted_activations = weights[:, :, None, None] * activations
        #  取消 梯度 相乘！！！
        cam = activations.sum(axis=1)
        return cam


    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations] 
        
        grads_list = [g.cpu().data.numpy()for g in self.activations_and_grads.gradients] 
        
        target_size = self.get_target_width_height(input_tensor)
 

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        # For Siamese Network
        for i in range(len(activations_list)):
            try:
                target_layer = self.target_layers[i]
            except:
                target_layer = self.target_layers[0]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
 
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            cam = np.float32(cam)
            scaled = scale_cam_image(cam, target_size)
 
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    
class SemanticSegmentationTarget:
    def __init__(self, category, mask=None):
        self.category = category
        # self.mask = torch.from_numpy(mask)
        # if torch.cuda.is_available():
        #     self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ]).sum()


def single_gpu_test_for_feature(model,data_loader,eval_image_indexs,target_layers):

    # # clean gpu memory when starting a new evaluation.
    # torch.cuda.empty_cache()
    
    model.eval()
    features = []
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, input_tensor in zip(loader_indices, data_loader):
        # print("batch_indices: ",batch_indices)
        if batch_indices[0] in eval_image_indexs: 
            with torch.no_grad():                 
                # target_layers = [model.module.decode_head.pre_process, model.module.decode_head.conv_seg] 
                targets = [SemanticSegmentationTarget(0)] # 1, 0 
                with MMGradCAM(model=model,
                        target_layers=target_layers,
                        use_cuda=torch.cuda.is_available(),
                        use_siam_layer=True) as cam:
                    grayscale_cam,result = cam(input_tensor=input_tensor,
                                        targets=targets) 
                    features.append(grayscale_cam)
                    results.extend(result)
        else:
            features.append([])
            results.extend(['result'])

        prog_bar.update() 

    return features,results



  
 