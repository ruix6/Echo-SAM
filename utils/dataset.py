import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, mode='train', num_points=5, size=512, aug=True):
        self.data_dir = data_dir
        self.mode = mode
        self.num_points = num_points
        self.img_paths, self.mask_paths = self.load_paths()
        self.resize1 = transforms.Resize((size, size))
        self.resize2 = transforms.Resize((size//4, size//4))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.augmentation = transforms.Compose(
            [
                transforms.GaussianBlur(3),
                transforms.RandomErasing()
            ]
        )
        self.aug = aug

    def load_paths(self):
        img_dir = os.path.join('/workspace/ussam/database', self.data_dir, self.mode, 'image')
        mask_dir = os.path.join('/workspace/ussam/database', self.data_dir, self.mode, 'mask')
        img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        mask_paths = [os.path.join(mask_dir, filename) for filename in os.listdir(mask_dir)]
        img_paths.sort()
        mask_paths.sort()
        return img_paths, mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #assert img_path.split('\\')[-1] == mask_path.split('\\')[-1], f'img:{img_path}, mask:{mask_path}Image and mask names do not match'

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img = self.resize1(img)
        img = self.to_tensor(img)
        if self.mode == 'train' and self.aug:
            img = self.augmentation(img)

        #img = self.normalize(img)
        mask = self.resize1(mask)
        low_mask = self.resize2(mask)

        mask = np.array(mask) / 255  # Convert to binary mask
        bbox = self.get_bbox(mask).unsqueeze(0)
        points_prompts, points_labels = self.get_points(mask, num_points=self.num_points)

        mask = self.to_tensor(mask)
        low_mask = self.to_tensor(low_mask)
        return {
            'image': img,
            'mask': mask,
            'low_mask': low_mask,
            'bbox': bbox,
            'points_prompts': points_prompts,
            'points_labels': points_labels
        }

    def get_bbox(self, mask, offset=5):
        mask_bool = mask.astype(bool)
        y_indices, x_indices = np.where(mask_bool)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, offset))
        x_max = min(W, x_max + np.random.randint(0, offset))
        y_min = max(0, y_min - np.random.randint(0, offset))
        y_max = min(H, y_max + np.random.randint(0, offset))
        return torch.tensor([x_min, y_min, x_max, y_max])

    def get_points(self, mask, num_points=1):
        mask_bool = mask.astype(bool)
        coords = np.argwhere(mask_bool)
        coords = coords[:, ::-1]
        points_prompts = coords[np.random.choice(len(coords), num_points, replace=False)]
        points_labels = np.ones(num_points)  # Assign label 1 for all points
        return torch.tensor(points_prompts), torch.tensor(points_labels)

class SegmentationDatasetCAMUS(Dataset):
    def __init__(self, data_dir, pos='LA', mode='train', num_points=5, size=512):
        self.data_dir = data_dir
        self.mode = mode
        self.mask_name = 'mask_'+pos
        self.num_points = num_points
        self.img_paths, self.mask_paths = self.load_paths()
        self.resize1 = transforms.Resize((size, size))
        self.resize2 = transforms.Resize((size//4, size//4))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.augmentation = transforms.Compose(
            [
                transforms.GaussianBlur(3),
                transforms.RandomErasing()
            ]
        )

    def load_paths(self):
        img_dir = os.path.join('/workspace/ussam/database', self.data_dir, self.mode, 'image')
        mask_dir = os.path.join('/workspace/ussam/database', self.data_dir, self.mode, self.mask_name)
        img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        mask_paths = [os.path.join(mask_dir, filename) for filename in os.listdir(mask_dir)]
        img_paths.sort()
        mask_paths.sort()
        return img_paths, mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #assert img_path.split('\\')[-1] == mask_path.split('\\')[-1], f'img:{img_path}, mask:{mask_path}Image and mask names do not match'

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img = self.resize1(img)
        img = self.to_tensor(img)
        if self.mode == 'train':
            img = self.augmentation(img)

        #img = self.normalize(img)
        mask = self.resize1(mask)
        low_mask = self.resize2(mask)

        mask = np.array(mask) / 255  # Convert to binary mask
        bbox = self.get_bbox(mask).unsqueeze(0)
        points_prompts, points_labels = self.get_points(mask, num_points=self.num_points)

        mask = self.to_tensor(mask)
        low_mask = self.to_tensor(low_mask)
        return {
            'image': img,
            'mask': mask,
            'low_mask': low_mask,
            'bbox': bbox,
            'points_prompts': points_prompts,
            'points_labels': points_labels
        }

    def get_bbox(self, mask, offset=5):
        mask_bool = mask.astype(bool)
        y_indices, x_indices = np.where(mask_bool)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, offset))
        x_max = min(W, x_max + np.random.randint(0, offset))
        y_min = max(0, y_min - np.random.randint(0, offset))
        y_max = min(H, y_max + np.random.randint(0, offset))
        return torch.tensor([x_min, y_min, x_max, y_max])

    def get_points(self, mask, num_points=5):
        mask_bool = mask.astype(bool)
        coords = np.argwhere(mask_bool)
        coords = coords[:, ::-1]
        points_prompts = coords[np.random.choice(len(coords), num_points, replace=False)]
        points_labels = np.ones(num_points)  # Assign label 1 for all points
        return torch.tensor(points_prompts), torch.tensor(points_labels)