import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import math


mean = [0.4488288587982963, 0.4371381120257274, 0.4040372117187323]
std = [0.2841556154456293, 0.27009665280451317, 0.2920475073076829]

def extract_patches(image,patch_size=256,overlap=64):
    height,width,_ = image.shape
    stride = patch_size - overlap
    patches = []
    patch_positions = []
    for y in range(0,height-patch_size+1,stride):
        for x in range(0,width-patch_size+1,stride):
            patch = image[y:y+patch_size,x:x+patch_size,:]
            patches.append(patch)
            patch_positions.append((x,y))
    return patches,patch_positions

def bicubic_downsample(image, scale_factor):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    width, height = image.size
    lr_image = image.resize((width // scale_factor, height // scale_factor), Image.BICUBIC)
    return lr_image

def load_hr_image(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            files.append(img_path)
    random_img = random.choice(files)
    image = cv2.imread(random_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    return image

def load_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def vis_patches(hr_image, patch_size=128,overlap = 32):
    patches,positions = extract_patches(hr_image,patch_size,overlap)
    stride = patch_size - overlap
    height,width,_ = hr_image.shape
    
    num_rows = (height - patch_size) // stride + 1
    num_cols = (width - patch_size) // stride + 1
    
    grid_height = num_rows * patch_size
    grid_width = num_cols * patch_size
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  
    for (y,x), patch in zip(positions,patches):
        grid_y = (y//stride)*patch_size
        grid_x = (x//stride)*patch_size
        if grid_y + patch_size <= grid_height and grid_x + patch_size <= grid_width:
            grid_img[grid_y:grid_y + patch_size, grid_x:grid_x + patch_size, :] = patch
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(hr_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(grid_img)
    axes[1].set_title("Extracted Patches")
    axes[1].axis("off")
    plt.show()
    
def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def super_resolve_patches(patches, model, device="cuda", vis=False, max_vis=5):
    model.to(device)
    model.eval()
    B, C, H, W, N = patches.shape 
    assert B == 1, "Batch size must be 1"
    sr_patches_list = []
    transform = transforms.Normalize(mean=mean, std=std)
    with torch.no_grad():
        for i in range(N):
            patch = patches[..., i]  
            patch = patch.squeeze(0)  
            patch = patch.float() / 255.0 
            patch_tensor = transform(patch).unsqueeze(0).to(device) 
            sr_patch_tensor = model(patch_tensor)  
            sr_patch_tensor = sr_patch_tensor.squeeze(0).cpu()
            sr_patch_tensor = denormalize(sr_patch_tensor,mean=mean,std=std)
            sr_patch = sr_patch_tensor.numpy().transpose(1, 2, 0)
            sr_patch = np.clip(sr_patch * 255.0, 0, 255).astype(np.uint8)
            sr_patches_list.append(sr_patch)

            # Visualization
            if vis and i < max_vis:
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                lr_debug = patch.cpu().numpy().transpose(1, 2, 0)
                axes[0].imshow(np.clip(lr_debug, 0, 1)) 
                axes[0].set_title("LR Patch")
                axes[0].axis("off")

                axes[1].imshow(sr_patch)
                axes[1].set_title("SR Patch")
                axes[1].axis("off")
                plt.show()
    sr_patches_np = np.stack(sr_patches_list, axis=0)
    sr_patches_torch = torch.from_numpy(sr_patches_np).float()
    sr_patches_torch = sr_patches_torch.permute(0, 3, 1, 2)
    sr_patches_torch = sr_patches_torch.unsqueeze(0).permute(0, 2, 3, 4, 1)
    return sr_patches_torch

def extract_patches_unfold(image,patch_size=128,hr_shape=None,overlap=32,scale=None):
    stride = patch_size - overlap
    H, W, C = image.shape
    if hr_shape is not None:
        H_hr, W_hr = hr_shape
        needed_lr_h = math.ceil(H_hr / scale)
        needed_lr_w = math.ceil(W_hr / scale)
        if H < needed_lr_h:
            extra_h = needed_lr_h - H
            image = np.pad(image, ((0, extra_h), (0, 0), (0, 0)), mode='constant')
            H += extra_h
        if W < needed_lr_w:
            extra_w = needed_lr_w - W
            image = np.pad(image, ((0, 0), (0, extra_w), (0, 0)), mode='constant')
            W += extra_w
            
    pad_bottom, pad_right = 0, 0
    
    if H < patch_size:
        pad_bottom = patch_size - H
    else:
        remainder_h = (H - patch_size) % stride
        if remainder_h != 0:
            pad_bottom = stride - remainder_h
    if W < patch_size:
        pad_right = patch_size - W
    else:
        remainder_w = (W - patch_size) % stride
        if remainder_w != 0:
            pad_right = stride - remainder_w
            
    image_padded = np.pad(
        image,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode='constant'
    )
    image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).unsqueeze(0).float()
    patches = F.unfold(image_tensor, kernel_size=patch_size, stride=stride)
    C = image_tensor.shape[1]
    num_patches = patches.shape[-1]
    patches = patches.view(1, C, patch_size, patch_size, num_patches)
    shape = (C, patch_size, patch_size, num_patches, pad_bottom, pad_right, H, W,hr_shape,scale)
    return patches,shape

def merge_patches_fold(sr_patches,lr_image_shape,patch_size,overlap):
    B, C, up_patch_size, up_patch_size2, N = sr_patches.shape
    assert B == 1 and up_patch_size == up_patch_size2
    _, _, _, _, pad_bottom, pad_right, H_orig, W_orig,hr_shape,scale = lr_image_shape
    lr_h_padded = H_orig + pad_bottom
    lr_w_padded = W_orig + pad_right
    hr_h_padded = lr_h_padded * scale
    hr_w_padded = lr_w_padded * scale
    sr_stride = (patch_size - overlap) * scale

    sr_patches_2d = sr_patches.view(B,C * up_patch_size * up_patch_size,N)
    sr_summed = F.fold(sr_patches_2d, output_size=(hr_h_padded, hr_w_padded),kernel_size=up_patch_size,stride=sr_stride)

    ones = torch.ones_like(sr_patches_2d)
    weight_map = F.fold(ones,output_size=(hr_h_padded, hr_w_padded),kernel_size=up_patch_size,stride=sr_stride)
    sr_stitched_padded = sr_summed / (weight_map + 1e-8)
    hr_h_original = H_orig * scale
    hr_w_original = W_orig * scale
    sr_stitched = sr_stitched_padded[..., :hr_h_original, :hr_w_original]
    if hr_shape is not None:
        H_hr, W_hr = hr_shape
        if sr_stitched.shape[-2] > H_hr:
            sr_stitched = sr_stitched[..., :H_hr, :]
        if sr_stitched.shape[-1] > W_hr:
            sr_stitched = sr_stitched[..., :,:W_hr]
    return sr_stitched


def visualize_patches_grid(patches, unfolded_shape):
    C, H_patch, W_patch, num_patches_total = unfolded_shape
    patches_np = patches.view(C, H_patch, W_patch, num_patches_total).permute(3, 1, 2, 0).numpy()
    patches_np = np.clip(patches_np, 0, 255).astype(np.uint8)
    grid_cols = math.ceil(math.sqrt(num_patches_total))  
    grid_rows = math.ceil(num_patches_total / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    axes = axes.flatten()  
    for i in range(num_patches_total):
        axes[i].imshow(patches_np[i])
        axes[i].set_title(f"Patch {i}")
        axes[i].axis("off")
    for i in range(num_patches_total, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
