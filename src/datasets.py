# src/datasets.py
from torch.utils.data import Dataset
import torch
import numpy as np
from src.io import load_case
from src.preprocess import make_4ch_slice

class BraTSSliceDataset(Dataset):
    def __init__(self, case_folders, transform=None, target_size=256, with_mask=True):
        self.cases = case_folders
        self.transform = transform
        self.target_size = target_size
        self.with_mask = with_mask
        self.index = []
        for c in self.cases:
            imgs, seg = load_case(c)
            Z = list(imgs.values())[0].shape[2]
            for z in range(Z):
                # optionally only add slices with tissue
                self.index.append((c,z))
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        c,z = self.index[idx]
        imgs, seg = load_case(c)
        x = make_4ch_slice(imgs, z, self.target_size)
        mask = None
        if self.with_mask and seg is not None:
            m = seg[:,:,z]
            m = cv2.resize(m, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            m = (m>0).astype('float32')
            mask = m[np.newaxis,...]
        x = x.astype('float32')
        return {'image':torch.from_numpy(x), 'mask': None if mask is None else torch.from_numpy(mask)}
