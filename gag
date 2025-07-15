# Deep Learning-Based Virtual Elastin Staining for VPI Assessment in NSCLC

This repository contains the code and resources for the research paper titled **"Deep Learning-Based Virtual Elastin Staining Improves Visceral Pleural Invasion Assessment in Lung Cancer"**. The project develops a conditional Generative Adversarial Network (cGAN)-based pipeline to generate synthetic Eosin-Based Elastin Fluorescence (EBEF) images from standard Hematoxylin and Eosin (H&E) stained whole-slide images (WSIs). This virtual elastin staining enhances the assessment of Visceral Pleural Invasion (VPI) in non-small cell lung cancer (NSCLC), reducing reliance on specialized stains like Victoria Blue.

## The preprocessing pipeline processes H&E and fluorescence WSIs to produce aligned image tiles through the following steps:
1. **Modified_Valis (Global Registration, Round 1)**: Performs global registration of H&E and fluorescence WSIs to align them at the slide level (workflow in `Modified_Valis/README.md`).
2. **Otsu Threshold Calculation**: Computes Otsu thresholds for `.svs` WSIs to guide tile extraction or segmentation (`calculate_otsu_threshold.py`).
3. **White Tile Removal**: Removes mostly white tiles from extracted image tiles based on thresholds (`remove_white_tiles.py`).
4. **Reinhard Normalization**: Normalizes H&E tiles to a target image’s color profile using Reinhard normalization (`normalize_images.py`).
5. **Z-Score Normalization**: Normalizes fluorescence tiles using per-channel Z-score normalization (`z_score_normalize_images.py`).
6. **SIFT Registration (Local Registration, Round 2)**: Performs local alignment of paired H&E and fluorescence tiles using the Scale-Invariant Feature Transform (SIFT) in ImageJ (workflow in `Registration_SIFT/README.md`).

The output is a set of co-registered H&E and fluorescence tiles ready for input into a GAN model to generate synthetic EBEF images for VPI assessment.

## Model options
The Pix2PixHD model uses the co-registered tiles from preprocessing to train or perform inference, generating synthetic EBEF images for VPI assessment. Model options are provided in the `pix2pixHD/` folder, with trained checkpoints available at [10.5281/zenodo.15881230](https://doi.org/10.5281/zenodo.15881230).

## Inference
1. **Background Replacement**: Standardizes backgrounds using a reference image, producing masks for the next step.
2. **Gamma Correction**: Normalizes intensities to a target average (130), aligning with Pix2PixHD input requirements.
3. **GAN model Inference**:  Feed gamma-corrected H&E tiles into the trained Pix2PixHD model to generate synthetic EBEF fluorescence tiles.
4. **White Tile Removal**: Removes mostly white tiles from extracted image tiles based on thresholds (`remove_white_tiles.py`).

## Postprocessing in Qupath
1. **Stiching_Tiles**: Stitches inferred tiles into a pyramidal OME-TIFF
2. **Combine_paired_Tiff**: combining images like he.tif and f.tif (fluorescence) into a single viewer display or a pyramidal OME-TIFF file.
