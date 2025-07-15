#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import morphology
import os
import argparse

def get_background_percentage_and_otsu(image_path, threshold_method, manual_threshold, otsu_offset, use_gaussian_blur, gaussian_blur_kernel):
    """
    Calculate the percentage of background pixels, Otsu threshold, and max between-class variance for an image.
    Applies otsu_offset when threshold_method is 'otsu'.

    Args:
        image_path (str): Path to the input image.
        threshold_method (str): Thresholding method ('otsu' or 'manual').
        manual_threshold (int): Manual threshold value for background detection.
        otsu_offset (int): Offset to adjust Otsu threshold (negative to preserve more tissue, positive for more background).
        use_gaussian_blur (bool): Whether to apply Gaussian blur before thresholding.
        gaussian_blur_kernel (tuple): Kernel size for Gaussian blur (width, height).

    Returns:
        tuple: (background_percentage, image_shape, otsu_threshold, max_variance) or (None, None, None, None) if image cannot be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None
    
    # Preprocess the image
    if use_gaussian_blur:
        processed_image = cv2.GaussianBlur(image, gaussian_blur_kernel, 0)
    else:
        processed_image = image

    # Convert to grayscale for intensity and thresholding
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Create a mask for background percentage calculation
    if threshold_method == "otsu":
        thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = max(0, min(255, thresh_val + otsu_offset))  # Apply offset and clamp
        _, background_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    elif threshold_method == "manual":
        _, background_mask = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Invalid threshold_method. Must be 'otsu' or 'manual'.")

    # Calculate background percentage
    total_pixels = background_mask.size
    background_pixels = np.sum(background_mask == 255)
    background_percentage = (background_pixels / total_pixels) * 100

    # Calculate Otsu threshold and maximum between-class variance for the image
    otsu_threshold = None
    max_variance = None
    if threshold_method == "otsu":
        otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold = max(0, min(255, otsu_threshold + otsu_offset))  # Apply offset and clamp
        # Compute histogram and between-class variance
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()  # Normalize histogram to probabilities
        w0 = np.cumsum(hist_norm)  # Cumulative weight for background
        w1 = 1 - w0  # Weight for foreground
        mu0 = np.cumsum(np.arange(256) * hist_norm) / (w0 + 1e-10)  # Mean of background
        mu1 = (np.cumsum((np.arange(256) * hist_norm)[::-1])[::-1]) / (w1 + 1e-10)  # Mean of foreground
        variance = w0 * w1 * (mu0 - mu1) ** 2  # Between-class variance
        max_variance = np.max(variance[np.isfinite(variance)])  # Maximum variance (bimodality measure)
        print(f"Image '{os.path.basename(image_path)}': Otsu threshold (adjusted) = {otsu_threshold:.2f}, "
              f"max variance = {max_variance:.2f}")
    
    return background_percentage, image.shape, otsu_threshold, max_variance

def select_otsu_image(image_dir, threshold_method, manual_threshold, otsu_offset, use_gaussian_blur, gaussian_blur_kernel):
    """
    Select an image with the highest histogram bimodality (max between-class variance) for Otsu threshold.

    Args:
        image_dir (str): Directory containing input images.
        threshold_method (str): Thresholding method ('otsu' or 'manual').
        manual_threshold (int): Manual threshold value.
        otsu_offset (int): Offset for Otsu threshold.
        use_gaussian_blur (bool): Whether to apply Gaussian blur.
        gaussian_blur_kernel (tuple): Kernel size for Gaussian blur.

    Returns:
        tuple: (selected_image_path, selected_otsu_threshold) or raises ValueError if no valid images.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not image_files:
        raise ValueError(f"No image files found in '{image_dir}'.")

    max_variance = -float('inf')
    selected_image_path = None
    selected_otsu_threshold = None

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        background_percentage, _, otsu_threshold, max_variance_val = get_background_percentage_and_otsu(
            image_path, threshold_method, manual_threshold, otsu_offset, use_gaussian_blur, gaussian_blur_kernel
        )
        if max_variance_val is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            continue
        if max_variance_val > max_variance:
            max_variance = max_variance_val
            selected_image_path = image_path
            selected_otsu_threshold = otsu_threshold

    if selected_image_path is None or selected_otsu_threshold is None:
        raise ValueError("No valid images found for Otsu threshold calculation.")
    
    print(f"Selected image for Otsu threshold: '{selected_image_path}' with max between-class variance {max_variance:.2f} "
          f"and adjusted Otsu threshold {selected_otsu_threshold:.2f}")
    return selected_image_path, selected_otsu_threshold

def select_reference_image(image_dir, threshold_method, manual_threshold, otsu_offset, use_gaussian_blur, gaussian_blur_kernel, background_threshold):
    """
    Select an image with the lowest background percentage.

    Args:
        image_dir (str): Directory containing input images.
        threshold_method (str): Thresholding method ('otsu' or 'manual').
        manual_threshold (int): Manual threshold value.
        otsu_offset (int): Offset for Otsu threshold.
        use_gaussian_blur (bool): Whether to apply Gaussian blur.
        gaussian_blur_kernel (tuple): Kernel size for Gaussian blur.
        background_threshold (float): Threshold for warning if lowest background percentage exceeds this value.

    Returns:
        tuple: (selected_image_path, selected_image_shape) or raises ValueError if no valid images.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not image_files:
        raise ValueError(f"No image files found in '{image_dir}'.")

    min_background_percentage = float('inf')
    selected_image_path = None
    selected_image_shape = None

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        background_percentage, image_shape, _, _ = get_background_percentage_and_otsu(
            image_path, threshold_method, manual_threshold, otsu_offset, use_gaussian_blur, gaussian_blur_kernel
        )
        if background_percentage is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            continue
        if background_percentage < min_background_percentage:
            min_background_percentage = background_percentage
            selected_image_path = image_path
            selected_image_shape = image_shape

    if selected_image_path is None:
        raise ValueError("No valid images found for background percentage calculation.")
    
    if min_background_percentage >= background_threshold:
        print(f"Warning: No image found with background percentage < {background_threshold}%. "
              f"Using image with lowest background percentage: {min_background_percentage:.2f}%.")
    
    print(f"Selected reference image: '{selected_image_path}' with background percentage {min_background_percentage:.2f}%")
    return selected_image_path, selected_image_shape

def process_image(image_path, ref_image_path, mask_output_path, ref_image_shape, threshold_method, manual_threshold, otsu_threshold, min_object_size, kernel_size, erosion_iterations, dilation_iterations, use_gaussian_blur, gaussian_blur_kernel):
    """
    Process a single image: identify the background, replace it with the reference image's background, overwrite the original, and save the mask.

    Args:
        image_path (str): Path to the input image.
        ref_image_path (str): Path to the reference image.
        mask_output_path (str): Path to save the binary mask.
        ref_image_shape (tuple): Shape of the reference image.
        threshold_method (str): Thresholding method ('otsu' or 'manual').
        manual_threshold (int): Manual threshold value.
        otsu_threshold (float): Precomputed Otsu threshold (used if threshold_method is 'otsu').
        min_object_size (int): Minimum size for objects to keep in the mask.
        kernel_size (tuple): Kernel size for morphological operations.
        erosion_iterations (int): Number of erosion iterations.
        dilation_iterations (int): Number of dilation iterations.
        use_gaussian_blur (bool): Whether to apply Gaussian blur.
        gaussian_blur_kernel (tuple): Kernel size for Gaussian blur.
    """
    # Load images
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}. Skipping.")
        return

    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise ValueError(f"Reference image not found at: {ref_image_path}")

    # Ensure reference image is the same size
    if ref_image.shape != image.shape:
        ref_image = cv2.resize(ref_image, (image.shape[1], image.shape[0]))

    # Preprocess the original image
    if use_gaussian_blur:
        processed_image = cv2.GaussianBlur(image, gaussian_blur_kernel, 0)
    else:
        processed_image = image

    # Create a mask where background is white (255)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    if threshold_method == "otsu":
        if otsu_threshold is None:
            thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_val = max(0, min(255, thresh_val + otsu_offset))  # Apply offset and clamp
        else:
            thresh_val = max(0, min(255, otsu_threshold))  # Use precomputed threshold
        _, background_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        print(f"Processing '{os.path.basename(image_path)}': Using adjusted Otsu threshold = {thresh_val:.2f}")
    else:
        _, background_mask = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)

    # Refine the background mask
    background_mask = morphology.remove_small_objects(background_mask.astype(bool), min_size=min_object_size).astype(np.uint8) * 255
    kernel = np.ones(kernel_size, np.uint8)
    if erosion_iterations > 0:
        background_mask = cv2.erode(background_mask, kernel, iterations=erosion_iterations)
    if dilation_iterations > 0:
        background_mask = cv2.dilate(background_mask, kernel, iterations=dilation_iterations)

    # Create tissue mask by inverting
    tissue_mask = cv2.bitwise_not(background_mask)

    # Combine parts from original and reference images
    original_tissue_part = cv2.bitwise_and(image, image, mask=tissue_mask)
    replacement_background_part = cv2.bitwise_and(ref_image, ref_image, mask=background_mask)
    combined_image = cv2.add(original_tissue_part, replacement_background_part)

    # Save the results
    cv2.imwrite(mask_output_path, background_mask)  # Save the background mask (background is white)
    cv2.imwrite(image_path, combined_image)  # Overwrite the original image

def main():
    """
    Main function to parse command-line arguments and run batch background replacement.
    """
    parser = argparse.ArgumentParser(description="Replace image backgrounds with a reference image's background and generate binary masks.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-mask-dir", type=str, required=True, help="Directory to save binary masks")
    parser.add_argument("--reference-image-path", type=str, default=None, help="Path to the reference image (default: auto-select image with lowest background percentage)")
    parser.add_argument("--threshold-method", type=str, choices=["otsu", "manual"], default="manual", help="Thresholding method ('otsu' or 'manual')")
    parser.add_argument("--manual-threshold", type=int, default=190, help="Manual threshold value (0-255) for background detection")
    parser.add_argument("--otsu-offset", type=int, default=25, help="Offset for Otsu threshold (negative to preserve more tissue, positive for more background)")
    parser.add_argument("--min-object-size", type=int, default=1000, help="Minimum size for objects to keep in the mask")
    parser.add_argument("--kernel-size", type=int, nargs=2, default=(7, 7), help="Kernel size for morphological operations (width height)")
    parser.add_argument("--erosion-iterations", type=int, default=3, help="Number of erosion iterations")
    parser.add_argument("--dilation-iterations", type=int, default=1, help="Number of dilation iterations")
    parser.add_argument("--use-gaussian-blur", action="store_true", help="Apply Gaussian blur before thresholding")
    parser.add_argument("--gaussian-blur-kernel", type=int, nargs=2, default=(5, 5), help="Kernel size for Gaussian blur (width height)")
    parser.add_argument("--background-threshold", type=float, default=1.0, help="Threshold for warning if lowest background percentage exceeds this value")
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.image_dir):
        print(f"Error: Input directory '{args.image_dir}' does not exist.")
        exit()

    # Create output mask directory
    os.makedirs(args.output_mask_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not image_files:
        print(f"No image files found in '{args.image_dir}'.")
        exit()

    print(f"Found {len(image_files)} images to process in '{args.image_dir}'.")

    # Select Otsu threshold if needed
    otsu_threshold = None
    if args.threshold_method == "otsu":
        try:
            _, otsu_threshold = select_otsu_image(
                args.image_dir, args.threshold_method, args.manual_threshold, args.otsu_offset,
                args.use_gaussian_blur, tuple(args.gaussian_blur_kernel)
            )
        except ValueError as e:
            print(f"Error: {e}")
            exit()

    # Select reference image
    reference_image_path = args.reference_image_path
    if reference_image_path is None:
        try:
            reference_image_path, ref_image_shape = select_reference_image(
                args.image_dir, args.threshold_method, args.manual_threshold, args.otsu_offset,
                args.use_gaussian_blur, tuple(args.gaussian_blur_kernel), args.background_threshold
            )
        except ValueError as e:
            print(f"Error: {e}")
            exit()
    else:
        ref_image = cv2.imread(reference_image_path)
        if ref_image is None:
            print(f"Error: Reference image not found at '{reference_image_path}'.")
            exit()
        ref_image_shape = ref_image.shape

    print(f"Starting batch processing of {len(image_files)} images in '{args.image_dir}'...")

    processed_count = 0
    for filename in image_files:
        try:
            print(f"Processing '{filename}'...")
            current_image_path = os.path.join(args.image_dir, filename)
            base_name, _ = os.path.splitext(filename)
            mask_filename = f"{base_name}.png"
            current_mask_path = os.path.join(args.output_mask_dir, mask_filename)

            process_image(
                current_image_path, reference_image_path, current_mask_path, ref_image_shape,
                args.threshold_method, args.manual_threshold, otsu_threshold, args.min_object_size,
                tuple(args.kernel_size), args.erosion_iterations, args.dilation_iterations,
                args.use_gaussian_blur, tuple(args.gaussian_blur_kernel)
            )
            processed_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {e}")

    print(f"\nBatch processing complete. Successfully processed {processed_count}/{len(image_files)} files.")
    print(f"Modified images have been saved back to: '{args.image_dir}'")
    print(f"Masks have been saved to: '{args.output_mask_dir}'")

if __name__ == "__main__":
    main()
