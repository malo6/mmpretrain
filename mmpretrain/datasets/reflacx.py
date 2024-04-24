# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mat4py
from mmengine import get_file_backend
import json
from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import DTD_CATEGORIES


@DATASETS.register_module()
class Reflacx(BaseDataset):
    """The Describable Texture Dataset (DTD).

    Args:
        data_root (str): The root directory for Describable Texture dataset.
    """  # noqa: E501


    def __init__(self, data_root: str, ann_file: str="reflacx.json",**kwargs):

        # /media/zihao/7f04820d-0354-4f38-a4b4-cd1064923632/data/reflacx-1.0.0

        self.backend = get_file_backend(data_root, enable_singleton=True)
        ann_file = self.backend.join_path(ann_file)

        super(Reflacx, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = json.load(open(self.ann_file))
        data_list = []
        for pair in pairs:
            mimic_id=pair['study_id']
            image_path=pair['image_path']
            reflacx_id=pair['reflacx_id']
            report=pair['report']
            img_path = self.backend.join_path(self.img_prefix, image_path)
            info = dict(img_path=img_path, report=report, reflacx_id=reflacx_id,mimic_id=mimic_id)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
