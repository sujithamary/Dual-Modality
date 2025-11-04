# src/visualize.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def token_attn_to_heatmap(attn, img_size=256, patch_size=16):
    # attn: [B, heads, Nq, Nk] - for visualize we can average heads and take mean over kv tokens
    # For MRI queries attending to HSI keys: attn averaged over heads and over query tokens -> attention per key token
    a = attn.mean(axis=1).mean(axis=2)  # [B, Nk]
    B, Nk = a.shape
    side = int(np.sqrt(Nk))
    heatmaps = []
    for b in range(B):
        h = a[b].reshape(side, side)
        h = cv2.resize(h, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        hm = (h - h.min()) / (h.max()-h.min()+1e-8)
        heatmaps.append(hm)
    return heatmaps

def overlay_heatmap_on_image(img, heatmap, alpha=0.5):
    # img: HxW or HxWx3
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cmap = plt.get_cmap('jet')
    hm_color = cmap(heatmap)[:, :, :3]
    hm_color = (hm_color*255).astype(np.uint8)
    blended = cv2.addWeighted(img.astype(np.uint8), 1-alpha, hm_color, alpha, 0)
    return blended
