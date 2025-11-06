from visualize import get_gradcam_mask, overlay_mask_on_image
import torch
from model_unet import get_unet
import cv2

model = get_unet()
model.load_state_dict(torch.load('checkpoints/best.pt', map_location='cpu'))

input_tensor = torch.randn(1,4,128,128) 
mask = get_gradcam_mask(model, input_tensor, device='cpu')

base_img = input_tensor[0,0].detach().numpy()
overlay = overlay_mask_on_image(base_img, mask)
cv2.imwrite('gradcam_overlay.png', overlay)
