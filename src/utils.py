from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.transforms.functional import center_crop, pil_to_tensor


def plot_tensor(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0))
    
def unpack_yolo_label(array: np.array):
    """Returns x_center, y_center, width, height from a yolo label array"""
    x, y, w, h = array[1:].ravel()
    return x, y, w, h
    
def yolo_bbox_relative_to_absolute_coords(x, y, w, h, W, H):
    """
    Args:
        x, y: patch center coordinates in relative coordinates
        w, h: patch width and height in relative coordinates
        W: image width
        H: image height
    Returns:
        x_center, y_center, width, height from the patch in absolute coordinates"""
    x = round(x * W)
    y = round(y * H)
    w = round(w * W)
    h = round(h * H)
    return x, y, w, h
    
def yolo_bbox_min_max_coords(x, y, w, h):
    """Returns x_min, x_max, y_min, y_max from absolute coordinates of
    a yolo bbox given by x_center, y_center, width, height"""
    x_min = x - w // 2
    x_max = x + w // 2
    y_min = y - h // 2
    y_max = y + h // 2
    return x_min, x_max, y_min, y_max
    
def get_patch_label(image: torch.Tensor, label: torch.Tensor, patch: Image) -> np.array:
    """Finds the most similar label to the patch by comparing their sizes."""
    
    patch_height, patch_width = pil_to_tensor(patch).shape[1:]
    H, W = image.shape[1], image.shape[2]

    if label.ndim == 1: # make those with a single bbox 2D so we can do same for loop for all labels
        label = label[np.newaxis, :]

    size_abs_diff = {}
    for i, l in enumerate(label):
        x, y, w, h = unpack_yolo_label(l)
        x, y, w, h = yolo_bbox_relative_to_absolute_coords(x, y, w, h, W, H)
        size_abs_diff[i] = abs(h - patch_height) + abs(w - patch_width)

    j = min(size_abs_diff, key=size_abs_diff.get) 
    return label[j]
    
