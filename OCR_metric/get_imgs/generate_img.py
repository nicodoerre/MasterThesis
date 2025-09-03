import argparse
import os
from instantiate_model import get_model
import argparse
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch
import sys
import subprocess
import kornia
import torchvision.transforms as transforms

def bicubic_downsample(image, scale_factor):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    width, height = image.size
    lr_image = image.resize((width // scale_factor, height // scale_factor), Image.BICUBIC)
    return lr_image

def super_resolve_full_image_esdr(lr_image_np, model, args, device="cuda"):
    model = model.to(device).eval()
    lr_tensor = to_tensor(lr_image_np).unsqueeze(0).to(device)      
    lr_tensor = lr_tensor * args.rgb_range                          
    with torch.no_grad():
        sr_tensor = model(lr_tensor)[0]                             
    sr_tensor = sr_tensor.clamp(0, args.rgb_range) / args.rgb_range
    sr_np = (sr_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0) \
            .round().astype(np.uint8)                               

    return sr_np

def super_resolve_full_image_wavemixsr(lr_image_np, model, device="cuda"):
    model = model.to(device).eval()
    lr = transforms.ToTensor()(lr_image_np).unsqueeze(0).to(device)  
    lr_ycbcr = kornia.color.rgb_to_ycbcr(lr)
    with torch.no_grad():
        sr_ycbcr = model(lr_ycbcr)          
    sr_rgb = kornia.color.ycbcr_to_rgb(sr_ycbcr).clamp(0, 1)
    sr_np = (sr_rgb.squeeze(0).permute(1,2,0).cpu().numpy() * 255).round().astype(np.uint8)
    return sr_np



def main(args):
    if args.random_img:
        image_folder = 'C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/HR'
        img_list = os.listdir(image_folder)
        rnd_img = np.random.choice(img_list)
        img_path = os.path.join(image_folder, rnd_img)
        rnd_img = cv2.imread(img_path)
        hr_img = cv2.cvtColor(rnd_img, cv2.COLOR_BGR2RGB)
        hr_img_shape = hr_img.shape
    else:
        hr_img_path = 'C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/HR/scatter_plot_20250519_082744_96d0780a.png'
        hr_img = cv2.imread(hr_img_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img_shape = hr_img.shape
        
    lr_img = bicubic_downsample(hr_img, args.scale)
    lr_image_np = np.array(lr_img)
    lr_path = 'C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/LR/lr_image.png' 
    Image.fromarray(lr_image_np).save(lr_path)
    
    if args.model_name == 'liif':
        resolution = f'{hr_img_shape[0]},{hr_img_shape[1]}'
        upscaled_path = 'C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/SR/sp2.png'
        cmd = [
            sys.executable, 'C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/LIIF_official/demo.py',
            '--input', lr_path,
            '--model', 'C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/LIIF_official/save/_train_custom/epoch-best.pth',
            '--resolution', resolution,
            '--output', upscaled_path,
            '--gpu', '0'
        ]
        subprocess.run(cmd, check=True)
        sr_img = np.array(Image.open(upscaled_path))
    elif args.model_name == 'wavemixsr':
        model = get_model(model_name=args.model_name, scale=args.scale)
        sr_img = super_resolve_full_image_wavemixsr(lr_image_np, model) 
        Image.fromarray(sr_img).save('C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/SR/sp2.png')
    else:
        model = get_model(model_name=args.model_name, scale=args.scale)
        sr_img = super_resolve_full_image_esdr(lr_image_np, model, args)
        Image.fromarray(sr_img).save('C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/SR/sp2.png')
    #Image.fromarray(sr_img).save('C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/OCR_metric/test_imgs/SR/sp2.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EDSR Model")
    parser.add_argument('--scale', type=int, default=4, help='Scale factor for image super-resolution')
    parser.add_argument('--model_name', type=str, default='edsr', help='Name of the model to instantiate (e.g., edsr, wavemixsr,liif)')
    parser.add_argument('--random_img' , action='store_true', help='Generate a random image for testing')
    parser.add_argument('--save_lr', action='store_true', help='Save the low-resolution image')
    parser.add_argument('--rgb_range', type=int, default=255, help='Range of RGB values (default: 255)')

    args = parser.parse_args()
    main(args)