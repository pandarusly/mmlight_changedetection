from mmseg.datasets.builder import PIPELINES
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets import DATASETS
from mmseg.datasets.pipelines import LoadAnnotations
from mmcd.datasets.custom import CDDataset



# @PIPELINES.register_module()
class LoadBinaryAnnotations(object):
    """Load annotations for change detection.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`opencd.CDDataset`. 

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged', # in opencd: grayscale 不能用在多类中
            backend=self.imdecode_backend).squeeze().astype(np.uint8) 
        # modify to format ann 
        if results['format_ann'] == 'binary':
            gt_semantic_seg_copy = gt_semantic_seg.copy()  
            gt_semantic_seg[gt_semantic_seg_copy >= 1] = 1
        else:
            raise ValueError('Invalid value {}'.format(results['format_ann']))
         
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@DATASETS.register_module()
class XiongAn_CD_Final_Dataset(CDDataset):

    CLASSES =('unchanged', 'buildings','buildozing','road')
    PALETTE =[[0, 0, 0],[31, 119, 180], [174, 199, 232], [255, 127, 14]]
    # 0: (0, 0, 0, 255),
    # 1: (31, 119, 180, 255),
    # 2: (174, 199, 232, 255),
    # 3: (255, 127, 14, 255),
    # 4: (255, 187, 120, 255),
    # 5: (44, 160, 44, 255),
    # 6: (152, 223, 138, 255),
    # 7: (214, 39, 40, 255),
    def __init__(self,classes= ('unchanged', 'buildings'),
            palette=[[255, 255, 255],[31, 119, 180]],**kwargs):
    
        super().__init__(
            sub_dir_1='A',
            sub_dir_2='B',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            **kwargs)
 
        
        self.format_ann = None 
        print("XiongAn_CD_Final_Dataset",self.label_map)

        self.gt_seg_map_loader = LoadAnnotations()
        # self.gt_seg_map_loader = MultiImgLoadAnnotations(
        # ) if gt_seg_map_loader_cfg is None else MultiImgLoadAnnotations(
        #     **gt_seg_map_loader_cfg)
        
    def results2img(self, results, imgfile_prefix, indices=None):
        """Write the segmentation results to images.
        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 4.
            # result = result * 255 # for binary change detection
            output = Image.fromarray(result.astype(np.uint8)).convert('P') 
            palette = np.zeros((len(self.PALETTE), 3), dtype=np.uint8)
            for label_id, color in enumerate(self.PALETTE):
                palette[label_id] = color 
            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files
    
    def format_results(self, results, imgfile_prefix, indices=None):
        """Format the results into dir (standard format for LoveDA evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, indices)
 
        return result_files

      
@DATASETS.register_module()
class XiongAn_CD_B_Dataset(XiongAn_CD_Final_Dataset):
    def __init__(self, classes=('unchanged', 'buildings'), palette=[[255, 255, 255], [31, 119, 180]], **kwargs):
        super().__init__(classes=classes, palette=palette, **kwargs)
        # LoadBinaryAnnotations pipline
        print("XiongAn_CD_Final_Dataset",self.label_map)
    def get_classes_and_palette(self, classes=None, palette=None): 
            if classes is None:
                self.custom_classes = False
                return self.CLASSES, self.PALETTE

            self.custom_classes = True
            if isinstance(classes, str):
                # take it as a file path
                class_names = mmcv.list_from_file(classes)
            elif isinstance(classes, (tuple, list)):
                class_names = classes
            else:
                raise ValueError(f'Unsupported type {type(classes)} of classes.')

            if self.CLASSES:
                if not set(class_names).issubset(self.CLASSES):
                    raise ValueError('classes is not a subset of CLASSES.')

                # dictionary, its keys are the old label ids and its values
                # are the new label ids.
                # used for changing pixel labels in load_annotations.
                self.label_map = {}
                for i, c in enumerate(self.CLASSES):
                    if c not in class_names:
                        self.label_map[i] = 0  # 原来是-1
                    else:
                        self.label_map[i] = class_names.index(c)

            palette = self.get_palette_for_custom_classes(class_names, palette)

            return class_names, palette    
    # def results2img(self, results, imgfile_prefix, indices=None):
    #     """Write the segmentation results to images.
    #     Args:
    #         results (list[ndarray]): Testing results of the
    #             dataset.
    #         imgfile_prefix (str): The filename prefix of the png files.
    #             If the prefix is "somepath/xxx",
    #             the png files will be named "somepath/xxx.png".
    #         indices (list[int], optional): Indices of input results, if not
    #             set, all the indices of the dataset will be used.
    #             Default: None.
    #     Returns:
    #         list[str: str]: result txt files which contains corresponding
    #         semantic segmentation images.
    #     """

    #     mmcv.mkdir_or_exist(imgfile_prefix)
    #     result_files = []
    #     for result, idx in zip(results, indices):

    #         filename = self.img_infos[idx]['filename']
    #         basename = osp.splitext(osp.basename(filename))[0]

    #         png_filename = osp.join(imgfile_prefix, f'{basename}.png')

    #         # The  index range of official requirement is from 0 to 4.
    #         # result = result * 255 # for binary change detection
    #         output = Image.fromarray(result.astype(np.uint8)).convert('L') 
             
    #         # img = Image.new("L",result.shape)
    #         # PALETTE = [0,0,0,255,0,0,0,255,0,255,165,0]
    #         # img.putpalette(PALETTE)
    #         # img.putdata(result.astype(np.uint8).flatten())
    #         # img.save(png_filename)
    #         output.putpalette(np.array(self.PALETTE))
    #         output.save(png_filename)
    #         result_files.append(png_filename)

    #     return result_files
    
    # def format_results(self, results, imgfile_prefix, indices=None):
    #     """Format the results into dir (standard format for LoveDA evaluation).
    #     Args:
    #         results (list): Testing results of the dataset.
    #         imgfile_prefix (str): The prefix of images files. It
    #             includes the file path and the prefix of filename, e.g.,
    #             "a/b/prefix".
    #         indices (list[int], optional): Indices of input results,
    #             if not set, all the indices of the dataset will be used.
    #             Default: None.
    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a list containing
    #             the image paths, tmp_dir is the temporal directory created
    #             for saving json/png files when img_prefix is not specified.
    #     """
    #     if indices is None:
    #         indices = list(range(len(self)))

    #     assert isinstance(results, list), 'results must be a list.'
    #     assert isinstance(indices, list), 'indices must be a list.'

    #     result_files = self.results2img(results, imgfile_prefix, indices)
 
    #     return result_files

