# src/datasets.py
from torch.utils.data import Dataset
import torch
import numpy as np
from io_utils import load_case
from preprocess import make_4ch_slice
import cv2

from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch
import os

class BraTSSliceDataset(Dataset):
    def __init__(self, case_dirs, with_mask=True):
        self.with_mask = with_mask
        self.samples = []
        for case_dir in case_dirs:
            flair = os.path.join(case_dir, f"{os.path.basename(case_dir)}_flair.nii")
            t1 = os.path.join(case_dir, f"{os.path.basename(case_dir)}_t1.nii")
            t1ce = os.path.join(case_dir, f"{os.path.basename(case_dir)}_t1ce.nii")
            t2 = os.path.join(case_dir, f"{os.path.basename(case_dir)}_t2.nii")
            if not all(os.path.exists(p) for p in [flair, t1, t1ce, t2]):
                print(f"⚠️ Skipping incomplete case: {case_dir}")
                continue
            seg = os.path.join(case_dir, f"{os.path.basename(case_dir)}_seg.nii") if with_mask else None
            self.samples.append((flair, t1, t1ce, t2, seg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            flair, t1, t1ce, t2, seg = self.samples[idx]
            imgs = [nib.load(p).get_fdata() for p in [flair, t1, t1ce, t2]]
            img = np.stack(imgs, axis=0).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-5)

            # take a random slice
            z = np.random.randint(0, img.shape[-1])
            img_slice = img[:, :, :, z]

            sample = {'image': torch.from_numpy(img_slice)}

            if self.with_mask and seg is not None:
                m = nib.load(seg).get_fdata()
                m_slice = (m[:, :, z] > 0).astype(np.float32)
                sample['mask'] = torch.from_numpy(m_slice[None, ...])
            else:
                sample['mask'] = torch.zeros((1, img.shape[1], img.shape[2]))

            return sample

        except Exception as e:
            print(f"⚠️ Skipping index {idx} due to error: {e}")
            return None
