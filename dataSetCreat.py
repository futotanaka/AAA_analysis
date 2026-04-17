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
        self.augment = augment
        self.aug_scale = aug_scale
        self.original_images = sorted(
            [os.path.join(root_dir, 'original', file)
             for file in os.listdir(os.path.join(root_dir, 'original'))], key=natural_sort_key
        )
        self.mask_images = sorted(
            [os.path.join(root_dir, 'masks', file)
             for file in os.listdir(os.path.join(root_dir, 'masks'))], key=natural_sort_key
        )

    def __len__(self):
        if self.augment:
            return int(len(self.original_images) * self.aug_scale)
        else:
            return len(self.original_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        augment_data = False
        if self.augment and idx >= len(self.original_images):
            # idx = random.choice(range(len(self.original_images)))
            idx = idx - len(self.original_images)
            augment_data = True
        
        original_path = self.original_images[idx]
        mask_path = self.mask_images[idx]
        original = np.load(original_path)
        mask = np.load(mask_path)
        
        mask_2channels = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        mask_2channels[0, :, :] = (mask == 1).astype(np.uint8)  # First channel for label 1
        mask_2channels[1, :, :] = (mask == 1) | (mask == 2).astype(np.uint8)  # Second channel for label 2
        
        # for softmax
        # mask_2channels = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        # mask_2channels[0, :, :] = (mask == 1).astype(np.uint8)  # First channel for label 1
        # mask_2channels[1, :, :] = (mask == 2).astype(np.uint8)  # Second channel for label 2
        # mask_2channels[2, :, :] = (mask == 0).astype(np.uint8)  # Third channel for background
        
        sample = {'original': original, 'mask': mask_2channels}
        if not augment_data:
            self.transform = transforms.Compose([ToTensor()])
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_first_item_dir(self):
        return self.original_images[0] # use with filename = os.path.basename(test_dataset.get_first_item_dir())
    
    def get_filename(self, idx):
        original_path = self.original_images[idx]
        mask_path = self.mask_images[idx]
        return os.path.basename(original_path), os.path.basename(mask_path)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        original, mask = sample['original'], sample['mask']

        if original.ndim == 3:  # If the input is 3D (3 slices)
            # Assuming the shape is (3, H, W), swap axes to (C, H, W)
            original = original.reshape(3, original.shape[1], original.shape[2])
        else:  # For single 2D slice
            # Assuming grayscale
            original = original.reshape(1, original.shape[0], original.shape[1])

        return {'original': torch.from_numpy(original).float(),
                'mask': torch.from_numpy(mask).float()}
        
class ApplyElasticTransform(object):
    """Apply ElasticTransform to both image and mask."""
    def __init__(self, alpha, sigma, interpolation=transforms.InterpolationMode.BILINEAR, fill=0):
        self.ETparas = transforms.ElasticTransform.get_params(alpha=[alpha,alpha], sigma=[sigma,sigma], size=[512, 512])

    def __call__(self, sample):
        original, mask = sample['original'], sample['mask']
        
        # Apply ElasticTransform to each channel separately
        # print(f'paras: {self.ETparas}')
        transformed_original = TF.elastic_transform(original,self.ETparas)
        transformed_mask = TF.elastic_transform(mask,self.ETparas)

        return {'original': transformed_original, 'mask': transformed_mask}
    
class RandomRotation(object):
    """Apply random rotation to both image and mask"""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        original, mask = sample['original'], sample['mask']
        
        rotated_original = TF.rotate(original, angle)
        rotated_mask = TF.rotate(mask, angle)

        return {'original': rotated_original, 'mask': rotated_mask}