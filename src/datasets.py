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
#dataset = 'mixed' # 'mixed' (bboxes), 'bildacher' (masks)

class MixedDataset(Dataset):
    
    dataset = 'mixed'

    def __init__(self):
        super().__init__()
        
        self.dataset_dir = root_data_dir / self.dataset
        self.images_dir = self.dataset_dir / 'images'
        self.images = sorted(os.listdir(self.images_dir))    
        
        # Transforms
        self.transforms = Compose([
            ToTensor(),
        ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images_dir / self.images[idx]
        image = Image.open(image_path)
        image = self.transforms(image)
        
        return image
    
    