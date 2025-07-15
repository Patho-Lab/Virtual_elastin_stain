# Deep Learning-Based Virtual Elastin Staining for VPI Assessment in NSCLC


This repository contains the code for the paper **"Deep Learning-Based Virtual Elastin Staining Improves Visceral Pleural Invasion Assessment in Lung Cancer"**.

This project introduces a deep learning pipeline to generate virtual elastin stains directly from standard Hematoxylin and Eosin (H&E) whole-slide images (WSIs). Accurate assessment of Visceral Pleural Invasion (VPI) in non-small cell lung cancer (NSCLC) is challenging on H&E alone due to poor elastin contrast. Our method leverages the intrinsic Eosin-Based Elastin Fluorescence (EBEF) from H&E slides to create a perfectly co-registered ground-truth dataset. This enables a conditional Generative Adversarial Network (cGAN) to perform a high-fidelity image-to-image translation, producing a synthetic EBEF image that enhances elastin visualization for pathologists.

The virtual stain significantly improves VPI assessment accuracy, offering a practical and validated framework to reduce reliance on costly and time-consuming special stains.

## Workflow Overview

The end-to-end pipeline consists of a preprocessing stage to create aligned training data, a training/inference stage using a cGAN model, and a post-processing stage to reconstruct the final image for diagnostic review.

![WORKFLOW](https://github.com/user-attachments/assets/dd633be9-81a4-4879-8da9-a01b7bfb4046)


## Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch, OpenSlide, Scikit-image, Pandas, NumPy, OpenCV, torchstain, Pillow
*   Docker
*   [Fiji/ImageJ](https://imagej.net/software/fiji/) with the "Linear Stack Alignment with SIFT" plugin.
*   [QuPath](https://qupath.github.io/) for post-processing and visualization.


## Usage

The workflow is divided into data preprocessing, model inference, and post-processing.

### 1. Data Preprocessing Pipeline

This pipeline processes paired H&E and fluorescence WSIs to produce precisely co-registered image tiles for training.

1.  **Global WSI Registration (`Modified_Valis`)**:
    Performs an initial, coarse alignment of the H&E and fluorescence WSIs at the slide level. See `Modified_Valis/README.md` for the detailed workflow.

2.  **Otsu Threshold Calculation (`calculate_otsu_threshold.py`)**:
    This script processes all `.svs` files in a directory to determine an optimal threshold for separating tissue from background. It creates a low-resolution thumbnail of each WSI, calculates the Otsu threshold, and saves the results to a CSV file.
    ```bash
    python calculate_otsu_threshold.py --input-dir /path/to/svs_files/ --output-csv /path/to/thresholds.csv
    ```

3.  **Tile Filtering (`remove_white_tiles.py`)**:
    This script identifies and removes image tiles (`.jpg`, `.png`) that consist mostly of background (white space). It uses the `thresholds.csv` file generated in the previous step to apply a custom white-pixel intensity threshold for each set of tiles.
    ```bash
    python remove_white_tiles.py --tiles-dir /path/to/extracted_tiles/ --csv-file /path/to/thresholds.csv
    ```

4.  **Color & Intensity Normalization**:
    *   **H&E Tiles (Reinhard) (`Reinhard_normalization.py`):** This script standardizes the color profile of H&E tiles using Reinhard color normalization from the `torchstain` library. It fits a normalizer to a user-provided `target-image` and applies it to all images found recursively in the `source-dir`.
        ```bash
        python Reinhard_normalization.py \
            --target-image /path/to/target_style.png \
            --source-dir /path/to/source_tiles/ \
            --output-dir /path/to/normalized_tiles/
        ```
    *   **Fluorescence Tiles (Z-Score) (`z_score_normalize_images.py`):** This script applies per-channel Z-score normalization to fluorescence image tiles. For each image, it independently normalizes the R, G, and B channels. The resulting values are then rescaled to the `[0, 255]` range to be saved as a standard image file, preserving the original folder structure.
        ```bash
        python z_score_normalize_images.py \
            --source-dir /path/to/fluorescence_tiles/ \
            --output-dir /path/to/normalized_fluorescence_tiles/
        ```

5.  **Local Tile Registration (`Registration_SIFT`)**:
    Performs fine-grained, local alignment on paired H&E and fluorescence tiles using SIFT in ImageJ. See `Registration_SIFT/README.md` for the workflow.

The output of this pipeline is a dataset of perfectly aligned H&E (source) and EBEF (target) tiles, ready for model training.

### 2. Model Training and Inference (Pix2PixHD)

The `pix2pixHD/` directory contains the code for training the model and performing inference.

#### Pre-trained Models
Trained model checkpoints are available on Zenodo:

**[https://doi.org/10.5281/zenodo.15881230](https://doi.org/10.5281/zenodo.15881230)**

We recommend using the **normalized pix2pixHD model trained on 1024×1024 pixel images** for inference.

#### Inference Steps
1.  **Extract Tiles**: First, extract tiles from your H&E WSI of interest (e.g., using QuPath).

2.  **Background Replacement (`background_replacement.py`)**:
    This script standardizes H&E tiles by replacing their background with a consistent texture from a single reference image. It overwrites the original tiles and generates corresponding binary tissue masks for later use.
    ```bash
    python background_replacement.py \
        --image-dir /path/to/inference_tiles/ \
        --output-mask-dir /path/to/generated_masks/
    ```

3.  **Gamma Correction (`gamma_correction.py`)**:
    This script standardizes the overall brightness of the H&E tiles by dynamically adjusting the gamma to reach a target average intensity (default: 130).
    ```bash
    python gamma_correction.py \
        --folder-path /path/to/inference_tiles/ \
        --target-avg 130
    ```

4.  **GAN Model Inference**:
    Feed the preprocessed H&E tiles into the trained Pix2PixHD model to generate synthetic EBEF tiles. Refer to the `pix2pixHD/` folder for specific commands.

5.  **Clean Synthetic Image Background (`remove_mask_area.py`)**:
    This is the final post-processing step for the generated synthetic EBEF tiles before stitching. It uses the binary masks created during the `background_replacement` step (Step 2) to clean up the GAN's output. For each synthetic tile, it replaces all non-tissue regions (as defined by the corresponding mask) with a uniform dark color. This removes any potential artifacts in the background and ensures a visually clean final WSI.

    **Important:** This script **overwrites the synthetic EBEF tiles** in place.

    ```bash
    python remove_mask_area.py \
        --input-folder /path/to/synthetic_EBEF_tiles/ \
        --mask-folder /path/to/generated_masks/
    ```

### 3. Post-processing in QuPath

The final steps involve reconstructing the generated tiles into a whole-slide format using the provided QuPath scripts.

1.  **Stitching Tiles (`Stiching_Tiles`)**:
    This script stitches the individual inferred synthetic EBEF tiles into a cohesive, pyramidal OME-TIFF WSI.

2.  **Combine Paired TIFFs (`Combine_paired_Tiff`)**:
    This script combines the original H&E WSI and the newly generated synthetic EBEF WSI into a single multi-channel file. This allows pathologists to toggle between the H&E and virtual elastin views seamlessly within QuPath.

## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{,
  title={Deep Learning-Based Virtual Elastin Staining Improves Visceral Pleural Invasion Assessment in Lung Cancer},
  author={},
  journal={},
  year={},
  volume={},
  pages={}
}
