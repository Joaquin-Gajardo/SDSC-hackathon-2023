import os
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import cv2
from PIL import Image

import torchvision
from torchvision.transforms import Compose, ToTensor, CenterCrop
from torch.utils.data import Dataset, DataLoader

from src import ROOT_DIR


DATA_DIR = ROOT_DIR / 'data' / 'raw'
OUT_IMAGE_RES = 1280


class MixedDataset(Dataset):

    dataset = 'mixed'

    def __init__(self):
        super().__init__()
        
        # Folder paths
        self.dataset_dir = DATA_DIR / self.dataset
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'yolo'
        self.patches_dir = DATA_DIR / 'patches'

        # File paths
        self.images = [path for path in self.images_dir.iterdir()]
        self.labels = [path for path in self.labels_dir.iterdir()]
        assert all([Path(path).exists() for path in self.labels]), 'Not all images have labels'
        
        # Get patches paths from this dataset (patches come from multiple datasets)
        self.patches = [self.patches_dir / str(path.name).replace('.png', '_rumex.png') for path in self.images]
    
        # Correct for missing patches (13 of them)
        if not all([Path(path).exists() for path in self.patches]):
            missing_patches = self.get_missing_patches()
            self.missing_patches_names = [path.name for path in missing_patches]
            self.patches = [path for path in self.patches if path.name not in self.missing_patches_names]
            self.images = [path for path in self.images if path.name.replace('.png', '_rumex.png') not in self.missing_patches_names]
            self.labels = [self.labels_dir / str(path.name).replace('png', 'txt') for path in self.images]
            assert len(self.images) == len(self.patches) == len(self.patches), 'Number of images, labels and patches files should be equal'
        
        # Define transforms
        self.transforms = Compose([
            ToTensor(),
            #CenterCrop(OUT_IMAGE_RES), # NOTE: If doing this we would need to exclude labels and patches that fall outside the crop
        ])

    def get_missing_patches(self):
        return [path for path in self.patches if not Path(path).exists()]
        
    def __len__(self):
        assert len(self.images) == len(self.labels) == len(self.patches)
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = np.loadtxt(self.labels[idx])
        patch = Image.open(self.patches[idx])
        
        # Apply transforms to images
        image = self.transforms(image)

        return image, label, patch
    
    
class BildacherBackgroundDataset(Dataset):
    
    dataset = 'bildacher'
    
    def __init__(self):
        super().__init__()
        
        # Folder paths
        self.dataset_dir = DATA_DIR / self.dataset
        self.background_images_dir = self.dataset_dir / 'no_objects'
        
        # File paths
        self.background_images = [path for path in self.background_images_dir.iterdir()]
        
        # Define transforms
        self.transforms = Compose([
            ToTensor(),
            CenterCrop(OUT_IMAGE_RES), # NOTE: when doing this remember to do this for MixedDataset as well so their images have the same size
        ])

    def __len(self):
        return len(self.background_images)
    
    def __getitem__(self, idx):
        image = Image.open(self.background_images[idx])
        image = self.transforms(image)
        return image

