# src/preprocess.py
import numpy as np
import cv2

def normalize_img(img):
    # zero-mean unit-std on non-zero voxels (BraTS has background zeros)
    mask = img > 0
    if mask.sum()==0:
        return img
    m = img[mask].mean()
    s = img[mask].std() + 1e-8
    out = (img - m) / s
    out[~mask] = 0
    return out

def make_4ch_slice(imgs, slice_idx, target_size=256):
    # imgs: dict of arrays per modality (H,W,Z)
    chans = []
    for mod in ['_flair.nii','_t1.nii','_t1ce.nii','_t2.nii']:
        arr = imgs[mod]
        sl = arr[:,:,slice_idx]
        sl = normalize_img(sl)
        sl = cv2.resize(sl, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        chans.append(sl)
    x = np.stack(chans, axis=0)  # [4,H,W]
    return x
