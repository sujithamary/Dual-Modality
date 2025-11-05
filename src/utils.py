# src/utils.py
import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    # pred and target in [B,1,H,W]
    pred_flat = pred.view(pred.shape[0], -1)
    t_flat = target.view(target.shape[0], -1)
    intersect = (pred_flat * t_flat).sum(1)
    denom = pred_flat.sum(1) + t_flat.sum(1)
    dice = (2*intersect + eps) / (denom + eps)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy(pred, target)
    d = dice_loss(pred, target)
    return bce_weight * bce + (1-bce_weight) * d

def dice_coeff(pred, target, thr=0.5, eps=1e-6):
    pred_bin = (pred > thr).float()
    pred_flat = pred_bin.view(pred.shape[0],-1)
    t_flat = target.view(target.shape[0],-1)
    intersect = (pred_flat * t_flat).sum(1)
    denom = pred_flat.sum(1) + t_flat.sum(1)
    dice = (2*intersect + eps) / (denom + eps)
    return dice.mean().item()
