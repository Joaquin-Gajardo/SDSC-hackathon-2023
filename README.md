# Alpine-Aster: MoreWeeds
Our project looks for leveraring generative AI for synthetizing data to ease the burder of the current extensive data collection necessary in agriculture. We focus on the problem of weed detection, whereby were localizing weeds with a drone is being done with object detection and semantic segmentation approaches for weed detection and subsequent killing with robots.

We prepare a demo dataset and trained a yolov8 model as a baseline.

We explored two ideas for data augmentation:
1. Stable diffusion inpainting to blend in some weeds patches extracted with existing bounding boxes labels, into a background image where annotators have not seen weeds.
2. GANs to generate new synthetic weeds images and to a similar process from above but without inpaitining.

## Setup
```
conda create -n hackathon python=3.11
conda activate hackathon
pip install -e .
```
## Prepare data for baseline
Running this [notebook](./notebook/prepare_data/pytorch_datasets_dev.ipynb)

## Running the code
Train baseline:
```
python src/train.py
```
Augmented training with synthetic dataset:
   1. [GAN training and data synthesis notebook](./train_gan.ipynb)
   2. [stable-diffusion-inpainting notebook](./notebook/eda/Image_Blending_Inpainting_2.ipynb)

## Results
- GANs:
  
    ![GANs](./results/gans/gan64.gif)

- Stable diffusion inpainting:
  
    ![Stable diffusion inpainting](./results/inpainting/Overlay_HuggingFace_Inpaint_difficult.png)


