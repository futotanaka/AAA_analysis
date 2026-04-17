# elastic deformation効果を確認するプログラム
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import imageio

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class CTImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False, aug_scale=1):
        """
        Args:
            root_dir (string): Root directory containing images; must include two subfolders: 'original' and 'masks'.
            transform (callable, optional): Optional preprocessing/transforms.
            augment (bool): Whether to perform data augmentation.
            aug_scale (int): Multiplicative factor for augmented data.
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
        # When augmentation is enabled and the index exceeds the number of original samples,
        # treat this sample as an augmented one
        if self.augment and idx >= len(self.original_images):
            idx = idx - len(self.original_images)
            augment_data = True
        
        original_path = self.original_images[idx]
        mask_path = self.mask_images[idx]
        original = np.load(original_path)
        mask = np.load(mask_path)
        
        mask_2channels = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        mask_2channels[0, :, :] = (mask == 1).astype(np.uint8)  # Channel 1: label 1
        mask_2channels[1, :, :] = ((mask == 1) | (mask == 2)).astype(np.uint8)  # Channel 2: labels 1 or 2
        
        sample = {'original': original, 'mask': mask_2channels}
        # For non-augmented data, directly use the default transform
        if not augment_data:
            self.transform = transforms.Compose([ToTensor()])
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_first_item_dir(self):
        return self.original_images[0]
    
    def get_filename(self, idx):
        original_path = self.original_images[idx]
        mask_path = self.mask_images[idx]
        return os.path.basename(original_path), os.path.basename(mask_path)

class ToTensor(object):
    """Convert numpy arrays in the sample to torch.Tensor."""
    def __call__(self, sample):
        original, mask = sample['original'], sample['mask']

        if original.ndim == 3:  # 3D input (e.g., 3 slices)
            original = original.reshape(3, original.shape[1], original.shape[2])
        else:  # 2D input: single channel
            original = original.reshape(1, original.shape[0], original.shape[1])

        return {'original': torch.from_numpy(original).float(),
                'mask': torch.from_numpy(mask).float()}

class ApplyElasticTransform(object):
    """Apply an elastic transform to both image and mask for geometric augmentation."""
    def __init__(self, alpha, sigma, interpolation=transforms.InterpolationMode.BILINEAR, fill=0):
        # Obtain parameters for the elastic transform (tune as needed)
        self.ETparas = transforms.ElasticTransform.get_params(alpha=[alpha, alpha], sigma=[sigma, sigma], size=[512, 512])

    def __call__(self, sample):
        original, mask = sample['original'], sample['mask']
        transformed_original = TF.elastic_transform(original, self.ETparas)
        transformed_mask = TF.elastic_transform(mask, self.ETparas)
        return {'original': transformed_original, 'mask': transformed_mask}

def tensor_to_np_img(img):
    """
    Convert a torch.Tensor or numpy array of shape (C, H, W) to (H, W) (grayscale)
    or (H, W, C) (RGB), and normalize to the range 0–255.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]  # Grayscale
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # RGB
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-8:
        norm_img = np.zeros_like(img)
    else:
        norm_img = (img - img_min) / (img_max - img_min) * 255
    return norm_img.astype(np.uint8)

def test_data_augmentation_effect(dataset, save_dir):
    """
    Test the effect of data augmentation (elastic deformation only):
    1. Fetch a raw (non-augmented) sample;
    2. Fetch a sample processed by the elastic transform (use index len(dataset.original_images) to force the augmentation path);
    3. Save both images to the target directory as original.png and augmented.png.
    
    Parameters:
        dataset: An instance of CTImagesDataset
        save_dir: Target directory for saving the images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Fetch a raw sample (augmentation disabled)
    dataset.augment = False
    dataset.transform = transforms.Compose([ToTensor()])
    test_id = 20
    sample_original = dataset[test_id]
    
    # Fetch an augmented sample: apply elastic deformation for data augmentation
    dataset.augment = True
    aug_transform = transforms.Compose([
         ToTensor(),
         ApplyElasticTransform(alpha=80.0, sigma=5.0, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    ])
    dataset.transform = aug_transform
    # Trigger the augmentation branch by using index len(dataset.original_images)
    sample_augmented = dataset[len(dataset.original_images)+test_id]
    
    # Convert tensors to numpy image arrays
    original_img_np = tensor_to_np_img(sample_original['original'])
    augmented_img_np = tensor_to_np_img(sample_augmented['original'])
    
    # Save images to the specified folder
    original_save_path = os.path.join(save_dir, 'original.png')
    augmented_save_path = os.path.join(save_dir, 'augmented.png')
    imageio.imwrite(original_save_path, original_img_np)
    imageio.imwrite(augmented_save_path, augmented_img_np)
    print(f"Img saved to {save_dir}")

if __name__ == '__main__':
    # Assume data are stored under the 'data' folder with two subfolders: 'original' and 'masks'
    data_root = '/home/sunjiawei/data/evar_ct_outputs/train'
    # Create the dataset instance (augmentation disabled by default)
    dataset = CTImagesDataset(root_dir=data_root, augment=False)
    # Directory to save test results
    save_folder = './figures/deformation'
    # Call the test function
    test_data_augmentation_effect(dataset, save_folder)
