# Deep Learning-Based Virtual Elastin Staining for VPI Assessment in NSCLC

[![Paper](https://img.shields.io/badge/paper-Scientific%20Publication-blue)](https://link-to-your-paper.com) <!---TODO: Add link to your published paper--->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15881230.svg)](https://doi.org/10.5281/zenodo.15881230)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
*   PyTorch, OpenSlide, Scikit-image, Pandas, NumPy, OpenCV, torchstain, Pillow
*   Docker
*   [Fiji/ImageJ](https://imagej.net/software/fiji/) with the "Linear Stack Alignment with SIFT" plugin.
*   [QuPath](https://qupath.github.io/) for post-processing and visualization.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup for Registration:**
    *   **VALIS (Global Registration):** Follow the setup instructions in `Modified_Valis/README.md`, which typically involves building or pulling a Docker container.
    *   **SIFT (Local Registration):** Ensure Fiji/ImageJ is installed and follow the setup guide in `Registration_SIFT/README.md`.

## Usage

The workflow is divided into data preprocessing, model inference, and post-processing.

### 1. Data Preprocessing Pipeline (for Training Data)

This pipeline processes paired H&E and fluorescence WSIs to produce precisely co-registered image tiles for training the GAN model.

1.  **Global WSI Registration (`Modified_Valis`)**:
    Performs an initial, coarse alignment of the H&E and fluorescence WSIs at the slide level.

2.  **Otsu Threshold Calculation (`calculate_otsu_threshold.py`)**:
    Processes all `.svs` files to determine an optimal threshold for separating tissue from background and saves the results to a CSV file.

3.  **Tile Filtering (`remove_white_tiles.py`)**:
    Removes image tiles that consist mostly of background using the thresholds from the previously generated CSV file.

4.  **Color & Intensity Normalization**:
    *   **H&E Tiles (`Reinhard_normalization.py`):** Standardizes the color profile of H&E tiles using Reinhard color normalization.
    *   **Fluorescence Tiles (`z_score_normalize_images.py`):** Applies per-channel Z-score normalization to fluorescence image tiles.

5.  **Local Tile Registration (`Registration_SIFT`)**:
    Performs fine-grained, local alignment on paired H&E and fluorescence tiles using SIFT in ImageJ.

### 2. Inference Pipeline (for Generating Virtual Stains)

This pipeline takes new H&E tiles, preprocesses them, and generates the final synthetic EBEF images.

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
@article{your_article_citation,
  title={Deep Learning-Based Virtual Elastin Staining Improves Visceral Pleural Invasion Assessment in Lung Cancer},
  author={Your, Name and et, al.},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XX-XX}
}
