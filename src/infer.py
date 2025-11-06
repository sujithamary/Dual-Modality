import os
import torch
import numpy as np
from io_utils import load_case
from preprocess import make_4ch_slice
from model_unet import get_unet
import cv2

def infer_case(case_folder, ckpt='checkpoints/best.pt', device='cuda'):
    imgs, seg = load_case(case_folder)
    model = get_unet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    Z = list(imgs.values())[0].shape[2]
    results = {}
    for z in range(Z):
        x = make_4ch_slice(imgs, z, target_size=256)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(xt)[0,0].cpu().numpy()  # [H,W]
        results[z] = pred
    return results, imgs, seg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    args = parser.parse_args()
    res, imgs, seg = infer_case(args.case, args.ckpt)
    
    best_z = max(res.keys(), key=lambda k: res[k].sum())
    pred = res[best_z]
    cv2.imwrite("pred_slice.png", (pred*255).astype('uint8'))
    print("Saved pred_slice.png for slice", best_z)