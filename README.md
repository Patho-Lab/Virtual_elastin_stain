# Deep Learning-Based Virtual Elastin Staining for VPI Assessment in NSCLC

This repository contains the code for the paper **"Deep Learning-Based Virtual Elastin Staining Improves Visceral Pleural Invasion Assessment in Lung Cancer"**.

This project introduces a deep learning pipeline to generate virtual elastin stains directly from standard Hematoxylin and Eosin (H&E) whole-slide images (WSIs). Accurate assessment of Visceral Pleural Invasion (VPI) in non-small cell lung cancer (NSCLC) is challenging on H&E alone due to poor elastin contrast. Our method leverages the intrinsic Eosin-Based Elastin Fluorescence (EBEF) from H&E slides to create a perfectly co-registered ground-truth dataset. This enables a conditional Generative Adversarial Network (cGAN) to perform a high-fidelity image-to-image translation, producing a synthetic EBEF image that enhances elastin visualization for pathologists.

The virtual stain significantly improves VPI assessment accuracy, offering a practical and validated framework to reduce reliance on costly and time-consuming special stains.

## Workflow Overview

The end-to-end pipeline consists of a preprocessing stage to create aligned training data, a training/inference stage using a cGAN model, and a post-processing stage to reconstruct the final image for diagnostic review.

![Workflow Diagram](https://raw.githubusercontent.com/your-username/your-repo/main/path/to/figure8.png)  
*Figure: The computational workflow, from WSI registration and tile preparation to model inference and final evaluation. (We recommend adding Figure 8 from your paper here for a visual guide).*

## Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Docker
*   [Fiji/ImageJ](https://imagej.net/software/fiji/) with the "Linear Stack Alignment with SIFT" plugin.
*   [QuPath](https://qupath.github.io/) for post-processing and visualization.



## Usage

The workflow is divided into data preprocessing, model inference, and post-processing.

### 1. Data Preprocessing Pipeline

This pipeline processes paired H&E and fluorescence WSIs to produce precisely co-registered image tiles for training.

1.  **Global WSI Registration (`Modified_Valis`)**:
    Performs an initial, coarse alignment of the H&E and fluorescence WSIs at the slide level. See `Modified_Valis/README.md` for the detailed workflow.

2.  **Otsu Threshold Calculation**:
    Computes an Otsu threshold from `.svs` WSIs to help identify and filter out tiles with low tissue content.
    ```bash
    python calculate_otsu_threshold.py --input_dir /path/to/svs_files/
    ```

3.  **Tile Filtering**:
    Removes tiles that are mostly background (white space) based on the calculated thresholds.
    ```bash
    python remove_white_tiles.py --tile_dir /path/to/tiles/ --threshold 220
    ```

4.  **Color & Intensity Normalization**:
    *   **H&E Tiles (Reinhard):** Standardizes the color profile of H&E tiles to match a target image.
        ```bash
        python normalize_images.py --source_dir /path/to/source_tiles/ --target_image /path/to/target.png
        ```
    *   **Fluorescence Tiles (Z-Score):** Normalizes fluorescence intensity using per-channel Z-scores.
        ```bash
        python z_score_normalize_images.py --input_dir /path/to/fluorescence_tiles/
        ```

5.  **Local Tile Registration (`Registration_SIFT`)**:
    Performs fine-grained, local alignment on paired H&E and fluorescence tiles using SIFT in ImageJ. See `Registration_SIFT/README.md` for the workflow.

The output of this pipeline is a dataset of perfectly aligned H&E (source) and EBEF (target) tiles, ready for model training.

### 2. Model Training and Inference (Pix2PixHD)

The `pix2pixHD/` directory contains the code for training the model and performing inference.

#### Pre-trained Models
Trained model checkpoints are available on Zenodo:

**[https://doi.org/10.5281/zenodo.15881230](https://doi.org/10.5281/zenodo.15881230)**

The model selected for final validation in the paper is the **normalized pix2pixHD model trained on 1024×1024 pixel images**. We recommend using this checkpoint for inference.

#### Inference Pipeline
To generate a synthetic EBEF image from a new H&E WSI:

1.  **Extract Tiles**: First, extract tiles from your H&E WSI of interest (e.g., using QuPath).

2.  **Background Replacement**:
    Standardizes the background of H&E tiles using a reference image. This script also generates masks required for later steps.

3.  **Gamma Correction**:
    Applies gamma correction to normalize tile intensities to a target average of 130, which was found to be optimal for our model.

4.  **GAN Model Inference**:
    Feed the preprocessed H&E tiles into the trained Pix2PixHD model to generate synthetic EBEF tiles. Refer to the `pix2pixHD/` folder for specific commands.

5.  **Filter Output Tiles**:
    Use `remove_white_tiles.py` to clean up any empty or near-empty tiles from the model's output.

### 3. Post-processing in QuPath

The final steps involve reconstructing the generated tiles into a whole-slide format using the provided QuPath scripts.

1.  **Stitching Tiles (`Stiching_Tiles`)**:
    This script stitches the individual inferred synthetic EBEF tiles into a cohesive, pyramidal OME-TIFF WSI.

2.  **Combine Paired TIFFs (`Combine_paired_Tiff`)**:
    This script combines the original H&E WSI and the newly generated synthetic EBEF WSI into a single multi-channel file. This allows pathologists to toggle between the H&E and virtual elastin views seamlessly within QuPath.

## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{your_article_citation,
  title={Deep Learning-Based Virtual Elastin Staining Improves Visceral Pleural Invasion Assessment in Lung Cancer},
  author={Chenglong Wang, e.t.,},
  journal={Journal Name},
  year={2024},
  volume={},
  pages={}
}
