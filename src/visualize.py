# src/visualize.py
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam_mask(model, input_tensor, target_mask=None, device='cuda'):
    # model: segmentation model; input_tensor: torch [1,4,H,W]
    model.eval()
    # pick final encoder layer depending on backbone - for resnet34: model.encoder.layer4
    target_layer = model.encoder.layer4
    cam = GradCAM(model=model, target_layers=[target_layer])
    if target_mask is None:
        # use mean of model output as target
        out = model(input_tensor.to(device))
        # segmentation target: class 1 for pixels with highest probability
        thr = 0.5
        mask = (out>thr).float()
    else:
        mask = target_mask
    targets = [SemanticSegmentationTarget(mask=mask.cpu().numpy()[0,0,:,:], category=0)]
    grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)
    # cam returns [B, H, W]
    return grayscale_cam[0]

def overlay_mask_on_image(img_chan, mask_prob, alpha=0.5):
    # img_chan: 2D float in [-3,3] normalized; mask_prob [H,W] in [0,1]
    img = img_chan.copy()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img_rgb = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap((mask_prob*255).astype('uint8'), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)
    return blended