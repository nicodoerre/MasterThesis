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


class MDSR(nn.Module):
    def __init__(self, num_filters=64, num_res_blocks=16, res_scale=0.1):
        super(MDSR, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        
        # Shared residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, res_scale) for _ in range(num_res_blocks)]
        )
        
        # Scale-specific upsampling branches
        self.upsample_x2 = self._make_upsample_branch(num_filters, scale_factor=2)
        self.upsample_x3 = self._make_upsample_branch(num_filters, scale_factor=3)
        self.upsample_x4 = self._make_upsample_branch(num_filters, scale_factor=4)
        
        # Output convolution layer
        self.conv3 = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)

    def _make_upsample_branch(self, num_filters, scale_factor):
        """Creates an upsampling branch for a given scale factor."""
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        )

    def forward(self, x, scale_factor=2):
        # Shared feature extraction
        x = self.conv1(x)
        x = self.res_blocks(x)
        
        # Scale-specific upsampling
        if scale_factor == 2:
            x = self.upsample_x2(x)
        elif scale_factor == 3:
            x = self.upsample_x3(x)
        elif scale_factor == 4:
            x = self.upsample_x4(x)
        else:
            raise ValueError("Scale factor must be 2, 3, or 4")
        
        # Final convolution layer
        x = self.conv3(x)
        return x
