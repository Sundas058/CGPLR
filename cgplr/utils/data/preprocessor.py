from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        item = self.dataset[index]
        if len(item) == 3:
            fname, pid, camid = item
            weight = None
        elif len(item) == 4:
            fname, pid, camid, weight = item
        else:
            raise ValueError('dataset item must be (fname, pid, camid) or (fname, pid, camid, weight)')
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if weight is None:
            return img, fname, pid, camid, index
        return img, fname, pid, camid, index, float(weight)
