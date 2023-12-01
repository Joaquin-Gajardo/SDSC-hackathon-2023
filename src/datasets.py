from pathlib import Path
from PIL import Image
import numpy as np

import torchvision
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Compose, ToTensor, CenterCrop
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src import ROOT_DIR
from src.utils import (
    is_bbox_outside_crop,
    get_patch_label,
    unpack_yolo_label,
    yolo_bbox_relative_to_absolute_coords,
    )


DATA_DIR = ROOT_DIR / 'data' / 'raw'
OUT_IMAGE_RES = 1280


class MixedDatasetPreprocessing(Dataset):

    dataset = 'mixed'

    def __init__(self, center_crop = True):
        super().__init__()
        
        # Folder paths
        self.dataset_dir = DATA_DIR / self.dataset
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'yolo'
        self.patches_dir = DATA_DIR / 'patches'
        #self.original_H, self.original_W = pil_to_tensor(Image.open(self.images[idx])).shape[1:]
        
        # File paths
        self.images = [path for path in self.images_dir.iterdir()]
        self.labels = [path for path in self.labels_dir.iterdir()]
        assert all([Path(path).exists() for path in self.labels]), 'Not all images have labels'
        
        # Get patches paths from this dataset (patches come from multiple datasets)
        self.patches = [self.patches_dir / str(path.name).replace('.png', '_rumex.png') for path in self.images]
        if not all([Path(path).exists() for path in self.patches]):
            self._extra_paths_setup()
            
        # Define transforms
        self.transforms = Compose([ToTensor()])
        
        # Discard labels outside crop boundaries and update images, labels and patches attributes accordingly (only 9 so we just discard them)
        if center_crop:
            self.patches_idxs_to_discard = self._patches_idxs_to_discard() # NOTE: requires iterating over the dataset
            self.images = [path for i, path in enumerate(self.images) if i not in self.patches_idxs_to_discard]
            self.labels = [path for i, path in enumerate(self.labels) if i not in self.patches_idxs_to_discard]
            self.patches = [path for i, path in enumerate(self.patches) if i not in self.patches_idxs_to_discard]

            # Now we can apply center crop form now on
            self.transforms.transforms.insert(len(self.transforms.transforms), CenterCrop(OUT_IMAGE_RES))

    def _patches_idxs_to_discard(self):
        """Rerturns a list of indexes of the patches ((bboxes) that fall outside the crop boundaries.
        This can be later also used for augmentation as well as the train set, only val set would be exluded."""
        patches_to_discard = []
        for i in range(len(self)):
            image, label, patch = self[i]
            
            # There are more than one bbox per image, we need to find which one is the one that we see in the patch
            # Finding closest match by patch size:  
            patch_height, patch_width = pil_to_tensor(patch).shape[1:]
            H, W = image.shape[1], image.shape[2]
            label_match = get_patch_label(image, label, patch)
            
            ## BBox coordinates
            x, y, w, h = unpack_yolo_label(label_match)
            x, y, w, h = yolo_bbox_relative_to_absolute_coords(x, y, w, h, W, H)
            if is_bbox_outside_crop(x, y, w, h, W, H, OUT_IMAGE_RES):
                patches_to_discard.append(i)
                
        return patches_to_discard
    
    def _extra_paths_setup(self):
        # Correct for missing patches (13 of them)
        if not all([Path(path).exists() for path in self.patches]):
            missing_patches = self._get_missing_patches()
            missing_patches_names = [path.name for path in missing_patches]
            self.patches = [path for path in self.patches if path.name not in missing_patches_names]
            self.images = [path for path in self.images if path.name.replace('.png', '_rumex.png') not in missing_patches_names]
            self.labels = [self.labels_dir / str(path.name).replace('png', 'txt') for path in self.images]
            assert len(self.images) == len(self.patches) == len(self.patches), 'Number of images, labels and patches files should be equal'
    
    def _get_missing_patches(self):
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

