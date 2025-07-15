#!/usr/bin/env python3

import torch
from torchvision import transforms
import torchstain
import cv2
import os
from pathlib import Path
import argparse

def normalize_images(target_image_path, source_dir, output_dir):
    """
    Normalize images in source_dir using Reinhard normalization based on a target image.

    Args:
        target_image_path (str): Path to the target image for normalization.
        source_dir (str): Directory containing images to normalize (recursively searched).
        output_dir (str): Directory to save normalized images, preserving subfolder structure.
    """
    # Validate inputs
    if not os.path.isfile(target_image_path):
        print(f"Error: Target image {target_image_path} does not exist or is not a file.")
        return
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory {source_dir} does not exist or is not a directory.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and convert target image to RGB
    target = cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB)
    if target is None:
        print(f"Error: Unable to read target image {target_image_path}.")
        return

    # Define transformation
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    # Initialize and fit Reinhard normalizer
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch', method='modified')
    try:
        normalizer.fit(T(target))
    except Exception as e:
        print(f"Error fitting Reinhard normalizer: {e}")
        return

    # Get list of all image files recursively
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [p for p in Path(source_dir).rglob('*') if p.suffix.lower() in image_extensions]
    total_images = len(image_files)

    if total_images == 0:
        print(f"No images found in {source_dir} with extensions {image_extensions}.")
        return

    print(f"Found {total_images} images to process.")

    # Process images with progress tracking
    for idx, image_path in enumerate(image_files, 1):
        # Read and convert image to RGB
        to_transform = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        if to_transform is None:
            print(f"Warning: Unable to read image at {image_path}")
            continue
        
        try:
            # Normalize
            t_to_transform = T(to_transform)
            norm = normalizer.normalize(I=t_to_transform)  # Reinhard doesn't return H, E
            
            # Ensure correct shape (C, H, W)
            if norm.shape[0] != 3:
                norm = norm.permute(2, 0, 1)
            
            # Convert to uint8
            norm_uint8 = norm.round().to(torch.uint8)
            
            # Convert to PIL Image
            norm_pil = transforms.ToPILImage()(norm_uint8)
            
            # Create output path, preserving subfolder structure
            relative_path = image_path.relative_to(source_dir)
            output_filename = f"{relative_path.stem}.jpg"
            output_path = Path(output_dir) / relative_path.parent / output_filename
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save normalized image
            norm_pil.save(output_path)
            print(f"\rProgress: {idx}/{total_images} - Saved {output_path}", end="", flush=True)
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue

    # Print newline after completion
    print("\nNormalization completed.")

def main():
    """
    Main function to parse arguments and run image normalization.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Normalize images using Reinhard normalization.")
    parser.add_argument(
        "--target-image",
        type=str,
        required=True,
        help="Path to the target image for normalization"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing images to normalize (searched recursively)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save normalized images"
    )
    args = parser.parse_args()

    # Run normalization
    normalize_images(args.target_image, args.source_dir, args.output_dir)

if __name__ == "__main__":
    main()
