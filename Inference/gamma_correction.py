#!/usr/bin/env python3

from PIL import Image
import numpy as np
import os
import argparse

def apply_gamma(image, gamma):
    """
    Apply gamma correction to an image using a lookup table.

    Args:
        image (PIL.Image): Input image in RGB mode.
        gamma (float): Gamma value to apply.

    Returns:
        PIL.Image: Gamma-adjusted image.
    """
    lut = [int(((i / 255.0) ** gamma) * 255.0 + 0.5) for i in range(256)]
    channels = image.split()
    adjusted_channels = [channel.point(lut) for channel in channels]
    return Image.merge('RGB', adjusted_channels)

def compute_gamma(original_avg, target_avg=128, tolerance=5):
    """
    Compute the gamma value to adjust the image’s average intensity toward the target.

    Args:
        original_avg (float): Original average pixel intensity.
        target_avg (float): Target average pixel intensity (default: 128).
        tolerance (float): Tolerance within which no adjustment is needed (default: 5).

    Returns:
        float: Computed gamma value (1.0 if no adjustment needed).
    """
    if abs(original_avg - target_avg) < tolerance:
        return 1.0  # No adjustment needed
    if original_avg < 1 or original_avg > 254:
        return 1.0  # Avoid extreme cases
    return np.log(target_avg / 255.0) / np.log(original_avg / 255.0)

def process_images(folder_path, target_avg=130, tolerance=5):
    """
    Adjust gamma of all images in a folder dynamically based on their original average intensity.

    Args:
        folder_path (str): Path to folder containing images.
        target_avg (float): Target average pixel intensity (default: 130).
        tolerance (float): Tolerance for no adjustment (default: 5).
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"No image files found in '{folder_path}'.")
        return

    print(f"Found {len(image_files)} images to process in '{folder_path}'.")

    processed_count = 0
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        try:
            # Open and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            original_array = np.array(image)
            
            # Calculate original average intensity
            original_avg = np.mean(original_array)
            
            # Compute dynamic gamma for this image
            gamma = compute_gamma(original_avg, target_avg, tolerance)
            
            # Apply gamma correction
            adjusted_image = apply_gamma(image, gamma)
            
            # Save the adjusted image (overwrites original)
            adjusted_image.save(image_path, quality=100)
            
            # Calculate and report the adjusted average
            adjusted_avg = np.mean(np.array(adjusted_image))
            print(f"{filename}: Original avg: {original_avg:.2f}, Gamma: {gamma:.2f}, Adjusted avg: {adjusted_avg:.2f}")
            processed_count += 1
        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

    print(f"\nProcessing complete. Successfully processed {processed_count}/{len(image_files)} files.")

def main():
    """
    Main function to parse command-line arguments and run gamma correction.
    """
    parser = argparse.ArgumentParser(description="Apply dynamic gamma correction to images in a folder to adjust average intensity.")
    parser.add_argument(
        "--folder-path",
        type=str,
        required=True,
        help="Directory containing input images (PNG, JPG, JPEG, BMP, GIF)"
    )
    parser.add_argument(
        "--target-avg",
        type=float,
        default=130,
        help="Target average pixel intensity (default: 130)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5,
        help="Tolerance for no adjustment (default: 5)"
    )
    args = parser.parse_args()

    process_images(args.folder_path, args.target_avg, args.tolerance)

if __name__ == "__main__":
    main()
