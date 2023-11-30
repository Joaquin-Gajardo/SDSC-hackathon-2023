import os
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import cv2
from PIL import Image

import torchvision
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader

from src import ROOT_DIR


DATA_DIR = ROOT_DIR / 'data' / 'raw'
#dataset = 'mixed' # 'mixed' (bboxes), 'bildacher' (masks)

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
        self.labels = [self.labels_dir / str(path.name).replace('png', 'txt') for path in self.images]
        # self.labels = [str(path) for path in self.images_dir.iterdir()]
        assert all([Path(path).exists() for path in self.labels]), 'Not all images have labels'
        
        # Get patches paths from this dataset
        self.patches = [self.patches_dir / str(path.name).replace('.png', '_rumex.png') for path in self.images]
        
        #assert not all([Path(path).exists() for path in self.patches]), 'Not all images have patches'
        
        # Define transforms
        self.transforms = Compose([
            ToTensor(),
        ])

    def get_missing_patches(self):
        return [path for path in self.patches if not Path(path).exists()]
        
    def __len__(self):
        assert len(self.images) == len(self.labels) == len(self.patches)
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images_dir / self.images[idx]
        image = Image.open(image_path)
        image = self.transforms(image)

        return image, self.labels[idx], self.patches[idx]

