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

root_data_dir = ROOT_DIR / 'data' / 'raw'

dataset = 'mixed' # 'mixed' (bboxes), 'bildacher' (masks)
dataset_dir = root_data_dir / dataset 

class MixedDataset(Dataset):
    dataset = 'mixed'

    def __init__(self, transforms = None):
        super().__init__()
        self.dataset_dir = root_data_dir / self.dataset
        self.images_dir = self.dataset_dir / 'images'
        self.images = sorted(os.listdir(self.images_dir))    
        
        # Transforms
        self.transforms = Compose([
            ToTensor(),
        ])

    def __getitem__(self, idx):
        image_path = self.images_dir / self.images[idx]
        # with open(image_path, 'rb') as img:
        #     image = f()
        if self.transforms is not None:
            if isinstance(tensor, list):
                tensor = [self.transforms(x) for x in tensor]
            else:
                tensor = self.transforms(tensor)
        
        return image
    
    def __len__(self):
        return len(self.images)
    