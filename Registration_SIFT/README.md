## Workflow for ImageJ SIFT Registration: Setting Paths and Placing Image Files

This workflow describes how to set directory paths and organize image files for the ImageJ script (`sift_registration.ijm`), which performs batch SIFT (Scale-Invariant Feature Transform) registration of paired Hematoxylin and Eosin (H&E) and fluorescence image tiles. The script fine-tunes local alignment of image pairs after VALIS-based global registration, producing co-registered images for training a conditional Generative Adversarial Network (cGAN) to generate synthetic Eosin-Based Elastin Fluorescence (EBEF) images for Visceral Pleural Invasion (VPI) assessment in non-small cell lung cancer (NSCLC).

### Prerequisites
- **Software**: ImageJ/Fiji with the Linear Stack Alignment with SIFT plugin installed, available from [https://imagej.net/Fiji](https://imagej.net/Fiji).
- **Hardware**: Workstation with sufficient RAM (e.g., 128 GB) to handle image stacks, ideally with an NVIDIA GPU for downstream cGAN processing.
- **Script**: The provided `sift_registration.ijm` script, with directory paths updated as described below.

### Workflow Steps

1. **Set Directory Paths in the Script**:
   - Open the `sift_registration.ijm` script in a text editor.
   - Locate the following lines in the "Main script" section:
     ```ijm
     heParent = "";
     ffParent = "";
     outputFolder = "";
     ```
   - Replace the empty paths with valid directory paths on your system. For example:
     ```ijm
     heParent = "/home/chan87/Desktop/training_elastic_second/he";
     ffParent = "/home/chan87/Desktop/training_elastic_second/ff";
     outputFolder = "/home/chan87/Desktop/training_elastic_second/tmp";
     ```
   - **Path Descriptions**:
     - `heParent`: The parent directory containing subfolders with H&E image tiles (e.g., `/home/chan87/Desktop/training_elastic_second/he`).
     - `ffParent`: The parent directory containing subfolders with fluorescence image tiles (e.g., `/home/chan87/Desktop/training_elastic_second/ff`).
     - `outputFolder`: The directory where aligned fluorescence images will be saved (e.g., `/home/chan87/Desktop/training_elastic_second/tmp`).
   - Ensure these directories exist on your system:
     ```bash
     mkdir -p /home/chan87/Desktop/training_elastic_second/he
     mkdir -p /home/chan87/Desktop/training_elastic_second/ff
     mkdir -p /home/chan87/Desktop/training_elastic_second/tmp
     ```
   - Save the updated script as `sift_registration.ijm` in your working directory (e.g., `/home/chan87/Desktop/training_elastic_second/`).

2. **Place Image Files**:
   - **H&E Images**:
     - Store H&E image tiles in subfolders under `heParent` (e.g., `/home/chan87/Desktop/training_elastic_second/he`).
     - Each subfolder should be named with the pattern `heXX` (e.g., `he01`, `he02`), where `XX` is a unique identifier.
     - Within each subfolder, place JPEG image files with filenames starting with the subfolder name followed by an underscore (e.g., `he01_image1.jpg`, `he01_image2.jpg` in `he01/`).
     - Example:
       ```
       /home/chan87/Desktop/training_elastic_second/he/
       ├── he01/
       │   ├── he01_image1.jpg
       │   ├── he01_image2.jpg
       ├── he02/
       │   ├── he02_image1.jpg
       ```
   - **Fluorescence Images**:
     - Store fluorescence image tiles in subfolders under `ffParent` (e.g., `/home/chan87/Desktop/training_elastic_second/ff`).
     - Each subfolder should be named with the pattern `fXX`, corresponding to the H&E subfolder `heXX` (e.g., `f01` pairs with `he01`).
     - Within each subfolder, place JPEG image files with filenames matching the H&E images, starting with the subfolder name followed by an underscore (e.g., `f01_image1.jpg`, `f01_image2.jpg` in `f01/`).
     - Example:
       ```
       /home/chan87/Desktop/training_elastic_second/ff/
       ├── f01/
       │   ├── f01_image1.jpg
       │   ├── f01_image2.jpg
       ├── f02/
       │   ├── f02_image1.jpg
       ```
   - **Image Requirements**:
     - Images should be in JPEG format, typically 1024x1024 pixels, derived from WSIs at 20x magnification (0.1760 µm/pixel for H&E).
     - Filenames must follow the pattern `<subfolder>_<common_part>.jpg`, where the `common_part` matches between H&E and fluorescence images (e.g., `he01_image1.jpg` pairs with `f01_image1.jpg`).
   - **Output Directory**:
     - The `outputFolder` (e.g., `/home/chan87/Desktop/training_elastic_second/tmp`) will store aligned fluorescence images with the prefix `aligned_` (e.g., `aligned_f01_image1.jpg`).
     - Ensure the directory exists or the script will create it automatically.

3. **Run the ImageJ Script**:
   - Open Fiji and load the updated `sift_registration.ijm`:
     - Go to `File > Open` and select the script, or drag and drop it into the Fiji window.
   - Execute the script:
     ```bash
     # Within Fiji
     run("Macro...", "open=/home/chan87/Desktop/training_elastic_second/sift_registration.ijm");
     ```
   - Alternatively, run in headless mode (if supported):
     ```bash
     fiji --headless --console -macro /home/chan87/Desktop/training_elastic_second/sift_registration.ijm
     ```
   - The script will:
     - Scan H&E subfolders under `heParent` and match with corresponding fluorescence subfolders under `ffParent`.
     - Filter images by prefix (e.g., `he01_`, `f01_`).
     - Align each H&E-fluorescence pair using the SIFT plugin with predefined parameters.
     - Save aligned fluorescence images as JPEGs in `outputFolder`.
     - Log progress and errors to the ImageJ console.

4. **Verify Outputs**:
   - Check the output directory (e.g., `/home/chan87/Desktop/training_elastic_second/tmp`) for aligned fluorescence images (e.g., `aligned_f01_image1.jpg`).
   - Review the ImageJ console log for errors, such as missing files or alignment failures.
   - Inspect aligned images in ImageJ or QuPath to confirm pixel-level alignment with H&E images.
