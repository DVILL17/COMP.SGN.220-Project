from typing import Tuple, Optional, Union, List
from pathlib import Path
import os

from torch.utils.data import Dataset
import numpy as np


import utils

class MyDataset(Dataset):
    def __init__(self, split: str = 'train') -> None:
        super().__init__()
        self.split = split
        self.features = []
        self.targets = []
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""
        features_dir = self.split + '_features'
        for f in utils.get_files_from_dir_with_pathlib(features_dir):
            vals = np.load(str(f), allow_pickle=True)
            self.features.append(vals['features'])
            self.targets.append(vals['targets'])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[item], self.targets[item]