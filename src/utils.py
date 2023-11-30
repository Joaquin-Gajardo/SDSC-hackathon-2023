import torch

from matplotlib import pyplot as plt

def plot_tensor(image_tensor) -> None:
    plt.imshow(image_tensor.permute(1, 2, 0))
    
    