import torch
import torch.nn.functional as F
from torchvision import models
import pytorch_msssim


def gradient_loss(sr,gt):
    channels = sr.shape[1]
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=sr.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)  
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)    
    grad_sr_x = F.conv2d(sr, sobel_x, padding=1, groups=channels)
    grad_sr_y = F.conv2d(sr, sobel_y, padding=1, groups=channels)
    grad_gt_x = F.conv2d(gt, sobel_x, padding=1, groups=channels)
    grad_gt_y = F.conv2d(gt, sobel_y, padding=1, groups=channels)
    loss_x = F.l1_loss(grad_sr_x, grad_gt_x)
    loss_y = F.l1_loss(grad_sr_y, grad_gt_y)
    return loss_x + loss_y


class PerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval().to(device)  # Use layers up to relu4_1
        for param in vgg.parameters():
            param.requires_grad = False  
        self.vgg = vgg
        self.criterion = torch.nn.L1Loss()
    def forward(self, sr, gt):
        sr_features = self.vgg(sr)
        gt_features = self.vgg(gt)
        return self.criterion(sr_features, gt_features)
    
def ssim_loss(sr, gt):
    return 1 - pytorch_msssim.ssim(sr, gt, data_range=1)

class Custom_Loss(torch.nn.Module):
    def __init__(self, device, alpha=1.0,beta=0.1,gamma=0.1,delta=0.1):
        super(Custom_Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device)
        self.alpha = alpha  
        self.beta = beta    
        self.gamma = gamma  
        self.delta = delta
    def forward(self,sr,gt,H=None, W=None):
        if sr.ndim == 3 and H is not None and W is not None:
            B = sr.shape[0]
            sr = sr.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            gt = gt.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        l1 = self.l1_loss(sr,gt)
        grad_loss = gradient_loss(sr,gt)
        perc_loss = self.perceptual_loss(sr,gt)
        ssim = ssim_loss(sr,gt)
        return self.alpha * l1 + self.beta * grad_loss + self.gamma * perc_loss + self.delta * ssim
        