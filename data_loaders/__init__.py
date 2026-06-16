import json
import os
from abc import abstractmethod
from typing import *

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        root (str): paths to the dataset
    """

    def __init__(self,
                 root: str,
                 instances: str= None,
                 ):
        super().__init__()
        self.root = root
        self.metadata = pd.DataFrame()

        self._stats = {}
        metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
        self._stats['Total'] = len(metadata)
        metadata, stats = self.filter_metadata(metadata, instances=instances)
        self._stats.update(stats)
        self._instances = metadata['sha256'].values.astype(np.bytes_)
        metadata.set_index('sha256', inplace=True)
        self.metadata = metadata
        self.dataset_params = json.load(open(os.path.join(root, 'dataset_params.json')))

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame, instances=None) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass

    @abstractmethod
    def get_instance(self, instance: str, index: int) -> Dict[str, Any]:
        pass

    def __len__(self):
        return len(self._instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            instance = self._instances[index].decode('UTF-8')
            return self.get_instance(instance, index)
        except Exception as e:
            # print(f"Instance {instance} raised Exception '{traceback.format_exc()}'. Resampling another instance.")
            print(f"Instance {instance} raised Exception '{e}'. Resampling another instance.")
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append('Root: ' + self.root)
        for k, v in self._stats.items():
            lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)

    @property
    def instances(self):
        instances = [self._instances[i].decode('UTF-8') for i in range(len(self._instances))]
        return instances