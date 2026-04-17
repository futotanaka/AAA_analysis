import os
import re
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class CTImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False, aug_scale=1):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.original_images = sorted(
            [os.path.join(root_dir, 'original', file)
             for file in os.listdir(os.path.join(root_dir, 'original'))], key=natural_sort_key
        )
        
    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        original_path = self.original_images[idx]
        original = np.load(original_path)
        
        sample = {'original': original}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_first_item_dir(self):
        return self.original_images[0] # use with filename = os.path.basename(test_dataset.get_first_item_dir())
    
    def get_filename(self, idx):
        original_path = self.original_images[idx]
        return os.path.basename(original_path)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        original = sample['original']

        if original.ndim == 3:  # If the input is 3D (3 slices)
            # Assuming the shape is (3, H, W), swap axes to (C, H, W)
            original = original.reshape(3, original.shape[1], original.shape[2])
        else:  # For single 2D slice
            # Assuming grayscale
            original = original.reshape(1, original.shape[0], original.shape[1])

        return {'original': torch.from_numpy(original).float()}