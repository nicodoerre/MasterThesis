import torch
import numpy as np
import kornia
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import ToTensor
import os



def load_image_kornia(path):
    '''Load image and convert to tensor suitable for Kornia processing.'''
    img = cv2.imread(path)  
    img_rgb = img[:, :, ::-1].copy()  
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0 
    return img_tensor


def extract_y(img_tensor):
    '''Extract Y channel from RGB image tensor using Kornia.'''
    ycbcr = kornia.color.rgb_to_ycbcr(img_tensor)
    return ycbcr[:, 0:1, :, :]


def compute_fft_kornia(y_tensor):
    '''Compute the 2D FFT and return the log-magnitude spectrum.'''
    y = y_tensor.squeeze(1)
    fft = torch.fft.fft2(y)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = torch.abs(fft_shifted)
    log_mag = torch.log1p(magnitude)
    return log_mag



def plot_fft_comparison(img_path,save_path=None):
    '''
    Plot FFT comparison between original and super-resolved images.
    '''
    img_tensor = load_image_kornia(img_path)
    y_tensor = extract_y(img_tensor)
    fft_mag = compute_fft_kornia(y_tensor)

    img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
    fft_np = fft_mag.squeeze().numpy()

    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(fft_np, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_fft(img_path, title, dx=0.0065, dy=0.0065, save_path=None):
    '''
    Plot the FFT of the Y channel of an image with frequency axes in cycles/mm.
    Parameters:
    - img_path: Path to the input image.
    - title: Title for the plot.
    - dx, dy: Pixel size in mm (default 0.0065 mm).
    - save_path: Optional path to save the plot.
    '''
    img_tensor = load_image_kornia(img_path)
    y_tensor = extract_y(img_tensor)   

    fft_log = compute_fft_kornia(y_tensor).squeeze().cpu().numpy()

    H, W = fft_log.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=dy))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=dx))

    plt.figure(figsize=(6, 6))
    plt.imshow(
        fft_log,
        cmap='gray',
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        origin='lower',
        aspect='equal'
    )
    plt.title(title if title else "FFT of Y Channel (log scale)")
    plt.xlabel(r'$f_x$ (cycles/mm)')
    plt.ylabel(r'$f_y$ (cycles/mm)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_raw_fft_magnitude(y_tensor):
    '''Compute the raw FFT magnitude of the Y channel tensor.'''
    y = y_tensor.squeeze(1) 
    fft = torch.fft.fft2(y)
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    mag = torch.abs(fft)
    return mag.squeeze().cpu().numpy()

def radial_profile(data, num_bins=100):
    '''Compute the radial profile of a 2D array.'''
    h, w = data.shape
    cy, cx = h//2, w//2
    Y, X = np.indices((h, w))
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).ravel()
    V = data.ravel()
    bins = np.linspace(0, R.max(), num_bins+1)
    inds = np.digitize(R, bins)
    prof = np.array([
        V[inds == i].mean() if np.any(inds == i) else np.nan
        for i in range(1, len(bins))
    ])
    # bin centers in pixels:
    bin_centers = (bins[:-1] + bins[1:]) * 0.5
    return bin_centers, prof

def plot_power_spectrum(gt_path, sr_path,save_path=None):
    '''Plot power spectrum of ground truth and super-resolved images.'''
    gt_tensor = load_image_kornia(gt_path)
    sr_tensor = load_image_kornia(sr_path)
    gt_y = extract_y(gt_tensor)
    sr_y = extract_y(sr_tensor)
    magnitude_gt = compute_raw_fft_magnitude(gt_y)
    magnitude_sr = compute_raw_fft_magnitude(sr_y)
    power_gt = magnitude_gt ** 2
    power_sr = magnitude_sr ** 2
    rad_px, power_gt_px = radial_profile(power_gt, num_bins=150)
    _, power_sr_px = radial_profile(power_sr, num_bins=150)
    W = magnitude_gt.shape[1]           
    freq = rad_px / W              
    mask = freq > 0                
    freq = freq[mask]
    
    power_gt_f = power_gt_px[mask]
    power_sr_f = power_sr_px[mask]
    amp_gt_f = np.sqrt(power_gt_f)
    amp_sr_f = np.sqrt(power_sr_f)
    
    plt.figure(figsize=(8,6),dpi=128)
    plt.loglog(freq, power_gt_f, label='GT',linewidth=2)
    plt.loglog(freq, power_sr_f, label='SR',linewidth=2, alpha=0.8)
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Power')
    plt.title('Radial Power Spectrum')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=128)
    plt.show()
    
def plot_power_spectrum_multi(gt_path, sr_path, save_path=None):
    '''Plot power spectrum of ground truth and multiple super-resolved images.'''

    gt_tensor = load_image_kornia(gt_path)
    gt_y = extract_y(gt_tensor)
    magnitude_gt = compute_raw_fft_magnitude(gt_y)
    power_gt = magnitude_gt ** 2
    rad_px, power_gt_px = radial_profile(power_gt, num_bins=150)
    W = magnitude_gt.shape[1]
    freq = rad_px / W
    mask = freq > 0
    freq = freq[mask]
    power_gt_f = power_gt_px[mask]

    if isinstance(sr_path, str):
        sr_items = [("SR", sr_path)]
    elif isinstance(sr_path, dict):
        sr_items = list(sr_path.items())
    else: 
        sr_items = []
        for p in sr_path:
            lbl = os.path.splitext(os.path.basename(p))[0] or "SR"
            sr_items.append((lbl, p))

    plt.figure(figsize=(8, 6), dpi=128)
    plt.loglog(freq,power_gt_f,label='Ground Truth',linewidth=3,color='black',linestyle='-',)

    for label, path in sr_items:
        sr_tensor = load_image_kornia(path)
        sr_y = extract_y(sr_tensor)
        magnitude_sr = compute_raw_fft_magnitude(sr_y)
        power_sr = magnitude_sr ** 2
        _, power_sr_px = radial_profile(power_sr, num_bins=150)
        power_sr_f = power_sr_px[mask]
        plt.loglog(freq, power_sr_f, label=label, linewidth=1, alpha=0.9)

    plt.xlabel('Spatial Frequency')
    plt.ylabel('Power')
    plt.title('Radial Power Spectrum')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=128)
    plt.show()

