import torch
from torch.utils.data import Dataset
import torch.nn as nn

from glob import glob
import pandas as pd
import numpy as np
import os.path as oph
from collections.abc import Sequence

from .transform import Compose
from .builder import DATASETS

@DATASETS.register_module()
class Icesat2ATL3(Dataset):
    def __init__(self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        filter_str = '*.csv',
        **kwargs
    ):
        super(Icesat2ATL3, self).__init__()
        self.data_root = data_root
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.aug_transform = None
        self.filter_str = filter_str
        self.data_files = self.get_datafiles()

    def get_datafiles(self):
        if isinstance(self.split, str):
            data_list = glob(oph.join(self.data_root, self.split, self.filter_str))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob(oph.join(self.data_root, split, "*"))
        else:
            raise NotImplementedError
        return data_list

    def prepare_train_data(self, idx):

        data_path = self.data_files[idx % len(self.data_files)]
        name = oph.basename(data_path)

        data = pd.read_csv(data_path)
        data_dict = {}

        data_dict['coords'] = data['Along_Track_Dist'].values[:, np.newaxis]
        data_dict['coords'] = (data_dict['coords'] - data_dict['coords'].min()) * 1e-3 ##convert to km
        data_dict['feature'] = data['Geoid_Corrected_Ortho_Height'].values[:, np.newaxis]
        data_dict['label'] = data['Manual_Label'].values
        data_dict['name'] = name

        return self.transform(data_dict)

    def __getitem__(self, idx):
        if not self.test_mode:
            return self.prepare_train_data(idx)
        else:
            return None

            # return data[['Along_Track_Dist','Geoid_Corrected_Ortho_Height']].values, data['Manual_Label'].values

    def __len__(self):
        return len(self.data_files) * self.loop
