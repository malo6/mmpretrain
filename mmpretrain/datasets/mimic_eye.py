# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mat4py
from mmengine import get_file_backend
import json
from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import DTD_CATEGORIES


@DATASETS.register_module()
class MIMIC_EYE(BaseDataset):
    """

    Args:
        data_root (str): The root directory for Describable Texture dataset.
    """  # noqa: E501


    def __init__(self, data_root: str='/public_bme/data/', ann_file: str = '/public_bme/data/mimic_eye.json', **kwargs):


        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.ann_file = self.backend.join_path(ann_file)



        super(MIMIC_EYE, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_root,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = json.load(open(self.ann_file))
        data_list = []
        for pair in pairs:
            mimic_id=pair['study_id']
            image_path=pair['image_path']
            source=pair['source']
            attention_path = pair['attention']
            report=pair['report']
            img_path = self.backend.join_path(self.img_prefix, image_path)           
            attention_path = self.backend.join_path(self.img_prefix, attention_path)

            info = dict(img_path=img_path, attention_path= attention_path,report=report, source=source,mimic_id=mimic_id)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
