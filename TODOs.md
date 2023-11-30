## Common
- [ ] Split `mixed` dataset in 80%/20% train and val, and create pytorch datasets that return annotation with corresponding patch (patches are from more datasets than just `mixed` so need to use the annotation to find the correct patches)
- [ ] Loader for `bildacher` no object images for background

## Baseline
- [ ] Create baseline object detection model, train on `mixed` dataset train split and test on the test split
- [ ] Calculate metrics: e.g. mean average precision (mAP)

## Data generation
### Idea 1: GAN training plants with black background (using `bildacher` dataset masks)
- [ ] Get masks from `bildacher` dataset and add black background
- [ ] Train unconditional GAN and sample X more images.
- [ ] Get new mask by color filtering or similar
- [ ] Impute the X images in different backgrounds from `bildacher` no objects images and get the new mask
- [ ] Extract bounding box?
- [ ] Create pytorch dataset with images and bounding boxes
- [ ] Combine dataset with `mixed` dataset training split and retrain baseline model
- [ ] Test on `mixed` dataset test split and compare metrics to baseline model

### Idea 2: imputing plant from patches in no object images and inpaiting to blend both backgrounds
- [ ] Get a list of all patches from train split of `mixed` dataset
- [ ] Function to inpute each patch on a different no object image at random location with a black padding background of X pixels around the patch, minding the borders.
- [ ] Pass each image through [stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) model (with diffusers) to get final augmentation dataset
- [ ] Write pytorch dataset for the inpainted dataset (with images and annotations)
- [ ] Combine datase with `mixed` dataset training split and retrain baseline model
- [ ] Test on `mixed` dataset test split and compare metrics to baseline model
