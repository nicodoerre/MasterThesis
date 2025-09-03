from PIL import Image
import random
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def bicubic_downsample(image, scale_factor):
    width, height = image.size
    lr_image = image.resize((width // scale_factor, height // scale_factor), Image.BICUBIC)
    return lr_image

def crop_patch(lr_image, hr_image, patch_size=48, scale_factor=2):
    lr_width, lr_height = lr_image.size
    hr_patch_size = patch_size * scale_factor 
    x = random.randint(0, lr_width - patch_size)
    y = random.randint(0, lr_height - patch_size)
    lr_patch = lr_image.crop((x, y, x + patch_size, y + patch_size))
    x_hr = x * scale_factor
    y_hr = y * scale_factor
    hr_patch = hr_image.crop((x_hr, y_hr, x_hr + hr_patch_size, y_hr + hr_patch_size))
    return lr_patch, hr_patch

