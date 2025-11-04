# src/model_unet.py
import segmentation_models_pytorch as smp
import torch.nn as nn

def get_unet(n_channels=4, encoder_name='resnet34', pretrained=True):
    model = smp.Unet(encoder_name=encoder_name, in_channels=n_channels, classes=1, activation='sigmoid', encoder_weights='imagenet' if pretrained else None)
    return model
