# VALIS Specification for Virtual Elastin Staining Pipeline

## Overview
VALIS (Virtual Alignment of pathology Image Series) is a Python library for registering and aligning whole-slide images (WSIs) in digital pathology. In this project, VALIS version 1.1.0 is used within a Docker container to perform hierarchical registration of brightfield Hematoxylin and Eosin (H&E) and fluorescence WSIs, ensuring pixel-level alignment for training a conditional Generative Adversarial Network (cGAN) to generate synthetic Eosin-Based Elastin Fluorescence (EBEF) images. This specification outlines the installation, configuration, usage, and modifications made to VALIS for the virtual elastin staining pipeline described in the research paper.

## VALIS Version
- **Version**: 1.1.0
- **Source**: [VALIS GitHub Repository](https://github.com/MathOnco/valis)
- **Purpose**: Aligns H&E and fluorescence WSIs to create co-registered image pairs for cGAN training in Visceral Pleural Invasion (VPI) assessment.

## Installation
VALIS 1.1.0 is installed and run within a Docker container for reproducibility.

## Modified Files
Three VALIS files were customized to optimize the registration pipeline for aligning H&E and fluorescence WSIs:

1. **preprocessing.py**:
   - **Purpose**: Enhances eosin channel extraction for H&E WSIs to align with fluorescence images.
   - **Modifications**:
     - Optimized `HEDeconvolution` class for eosin channel isolation.
     - Adjusted for specific staining protocols (no hematoxylin differentiation, 1-second eosin decolorization) to maximize elastin-to-collagen contrast (E/C ratio).
     - Added gamma correction for H&E standardization.

2. **registration.py**:
   - **Purpose**: Customizes registration logic for high-fidelity alignment of H&E and fluorescence WSIs.
   - **Modifications**:
     - Enhanced rigid registration to handle reflections between modalities.
     - Integrated `MicroRigidRegistrar` for precise local alignment.
     - Excluded non-rigid deformation to preserve tissue architecture.

3. **step1.py**:
   - **Purpose**: Orchestrates the VALIS registration workflow.
   - **Modifications**: Corrected for project-specific requirements (see below).
  
# Define input and output directories in **step1.py**
slide_src_dir = "/data/he"
results_dst_dir = "/data/af"
registered_slide_dst_dir = "/data/afr"
reference_slide = "he.tiff"


