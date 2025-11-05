from visualize import get_gradcam_mask, overlay_mask_on_image
import torch
from model_unet import get_unet
import cv2

# Load model
model = get_unet()
model.load_state_dict(torch.load('checkpoints/best.pt', map_location='cpu'))

# Load one test slice tensor [1,4,H,W]
input_tensor = torch.randn(1,4,128,128)  # example shape (replace with actual loaded MRI slice)
mask = get_gradcam_mask(model, input_tensor, device='cpu')

# Overlay GradCAM mask
base_img = input_tensor[0,0].detach().numpy()  # use first modality for visualization
overlay = overlay_mask_on_image(base_img, mask)
cv2.imwrite('gradcam_overlay.png', overlay)
