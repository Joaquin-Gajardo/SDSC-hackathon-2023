#%%
import os
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import cv2

from src import ROOT_DIR

#%%
# 1. Load COCO per dataset
# 2. make splits with cocosplit
# 3. Define dataset and dataloaders
# 4. Use the splits from coco to semantic masks as well for bildacher dataset 

#%%
root_data_dir = ROOT_DIR / 'data' / 'raw'
dataset = 'mixed' # 'mixed' (bboxes), 'bildacher' (masks)
dataset_dir = root_data_dir / dataset 

#%%
bboxes = dataset_dir / 'coco' / 'output.json'
images = dataset_dir / 'images'

#%%
coco = COCO(bboxes)

#%%
catIds = coco.getCatIds(catNms=['rumex_leaf']) #Add more categories ['person','dog']
imgIds = coco.getImgIds(catIds=catIds )
if not os.path.exists(seg_output_path):
    os.mkdir(original_img_path)
if not os.path.exists(original_img_path):
    os.mkdir(original_img_path)
# %%
