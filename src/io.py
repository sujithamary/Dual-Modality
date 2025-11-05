import nibabel as nib
import numpy as np
import os

MODS = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']

def load_case(folder):
    # folder contains files named like BraTS20_Training_001_flair.nii, etc.
    imgs = {}
    for suffix in MODS:
        f = None
        for fname in os.listdir(folder):
            if fname.endswith(suffix):
                f = os.path.join(folder, fname)
                break
        if f is None:
            raise FileNotFoundError(f"Missing modality {suffix} in {folder}")
        nim = nib.load(f)
        arr = nim.get_fdata().astype('float32')  # shape (H,W,Z)
        imgs[suffix] = arr
    # seg (optional)
    seg_path = None
    for fname in os.listdir(folder):
        if fname.endswith('_seg.nii') or fname.endswith('_seg.nii.gz'):
            seg_path = os.path.join(folder, fname); break
    seg = None
    if seg_path:
        seg = nib.load(seg_path).get_fdata().astype('uint8')
    return imgs, seg
