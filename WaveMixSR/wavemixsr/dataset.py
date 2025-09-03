from torch.utils.data import Dataset
from utils import crop_patch,bicubic_downsample
import kornia 
import torchvision.transforms as transforms
from PIL import Image

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_images, patch_size, scale_factor):
        self.hr_images = hr_images
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')

        lr_img  = bicubic_downsample(hr_img, self.scale_factor)
        lr, hr  = crop_patch(lr_img, hr_img, self.patch_size, self.scale_factor)

        lr, hr = self.to_tensor(lr), self.to_tensor(hr)
        lr, hr = kornia.color.rgb_to_ycbcr(lr), kornia.color.rgb_to_ycbcr(hr)
        return lr, hr