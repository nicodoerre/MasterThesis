import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_filters, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        residual = self.block(x)
        return x + residual * self.res_scale  # Apply residual scaling    
    
class UpsampleBlock(nn.Module):
    def __init__(self, num_filters, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.upsample(x)


class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_filters=64, num_res_blocks=16, res_scale=0.1):
        super(EDSR, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, res_scale) for _ in range(num_res_blocks)]
        )
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.upsample = UpsampleBlock(num_filters, scale_factor)
        self.conv3 = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = self.res_blocks(x)
        x = self.conv2(residual) + x  # Add skip connection
        x = self.upsample(x)
        x = self.conv3(x)
        return x