#!/usr/bin/env python3

import torch
from torchvision import transforms
import cv2
import os
from pathlib import Path
import argparse

def z_score_normalize(tensor):
    """
    Apply Z-score normalization per channel.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) with values in [0, 255].

    Returns:
        torch.Tensor: Z-score normalized tensor.
    """
    # Initialize output tensor
    norm_tensor = torch.zeros_like(tensor)
    
    # Process each channel (R, G, B)
    for c in range(tensor.shape[0]):
        channel = tensor[c]
        mean = channel.mean()
        std = channel.std()
        
        # Avoid division by zero
        if std > 0:
            norm_tensor[c] = (channel - mean) / std
        else:
            norm_tensor[c] = channel - mean  # If std is 0, just subtract mean
    
    return norm_tensor

def normalize_images(source_dir, output_dir):
    """
    Normalize images in source_dir using Z-score normalization, preserving subfolder structure.

    Args:
        source_dir (str): Directory containing images to normalize (searched recursively).
        output_dir (str): Directory to save normalized images.
    """
    # Validate source directory
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory {source_dir} does not exist or is not a directory.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define transformation to convert image to tensor
    T = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] float tensor
        transforms.Lambda(lambda x: x * 255)  # Scale to [0, 255]
    ])

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
        try:
            # Read and convert image to RGB
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            if image is None:
                print(f"Warning: Unable to read image at {image_path}")
                continue
            
            # Convert to tensor
            t_image = T(image)
            
            # Apply Z-score normalization
            norm_tensor = z_score_normalize(t_image)
            
            # Rescale to [0, 255] for saving as image
            norm_tensor = (norm_tensor - norm_tensor.min()) / (norm_tensor.max() - norm_tensor.min() + 1e-8) * 255
            norm_uint8 = norm_tensor.round().to(torch.uint8)
            
            # Ensure correct shape (C, H, W)
            if norm_uint8.shape[0] != 3:
                norm_uint8 = norm_uint8.permute(2, 0, 1)
            
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
    Main function to parse arguments and run Z-score normalization.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Normalize images using Z-score normalization.")
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
    normalize_images(args.source_dir, args.output_dir)

if __name__ == "__main__":
    main()
