from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.transforms.functional import center_crop


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

def is_bbox_outside_crop(x, y, w, h, W, H, crop_size):
    """
    Returns True if any point of the bounding box is outside the crop boundaries, otherwise False
    """
    # Calculate bounding box coordinates
    bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = yolo_bbox_min_max_coords(x, y, w, h)

    # Calculate crop boundaries
    crop_left = (W - crop_size) / 2
    crop_top = (H - crop_size) / 2
    crop_right = crop_left + crop_size
    crop_bottom = crop_top + crop_size

    # Check if any bounding box coordinate is outside the crop boundaries
    if bbox_x_min < crop_left or bbox_x_max > crop_right or bbox_y_min < crop_top or bbox_y_max > crop_bottom:
        return True  # Bounding box is partially or fully outside the crop
    else:
        return False  # Bounding box is completely inside the crop
    
def get_patch_label(image: torch.Tensor, label: torch.Tensor, patch: Image) -> np.array:
    """Finds the most similar label to the patch by comparing their sizes."""
    patch_height, patch_width = pil_to_tensor(patch).shape[1:]

    if label.ndim == 1: # make those with a single bbox 2D so we can do same for loop for all labels
        label = label[np.newaxis, :]

    size_abs_diff = {}
    for i, l in enumerate(label):
        x, y, w, h = unpack_yolo_label(l)
        x, y, w, h = yolo_bbox_relative_to_absolute_coords(x, y, w, h, W, H)
        size_abs_diff[i] = abs(h - patch_height) + abs(w - patch_width)

    j = min(size_abs_diff, key=size_abs_diff.get) 
    return label[j]
    
