#!/usr/bin/env python3

from PIL import Image
import numpy as np
import os
import argparse

def find_paired_files(input_folder, mask_folder):
    """
    Find paired image and mask files based on matching base filenames.

    Args:
        input_folder (str): Directory containing input images (JPEG or PNG).
        mask_folder (str): Directory containing mask images (JPEG or PNG).

    Returns:
        list: List of tuples (input_file, mask_file) for paired files.
    """
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.jpg', '.png'))]
    
    paired_files = []
    for mask_file in mask_files:
        mask_base = os.path.splitext(mask_file)[0]  # Remove extension
        for input_file in input_files:
            if mask_base in input_file:  # Check if mask base name is a substring of input filename
                paired_files.append((input_file, mask_file))
    
    return paired_files

def process_image_pair(input_path, mask_path, output_path, fill_value=(5, 5, 5)):
    """
    Process an image-mask pair: replace background (mask > 127) with a constant RGB fill value.

    Args:
        input_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        output_path (str): Path to save the processed image (overwrites input if same).
        fill_value (tuple): RGB value to fill background (default: (5, 5, 5)).

    Returns:
        bool: True if processing succeeds, False otherwise.
    """
    try:
        # Load the image and mask
        image = Image.open(input_path).convert('RGB')  # Convert to RGB
        mask = Image.open(mask_path).convert('L')      # Mask remains grayscale
        
        # Convert to numpy arrays
        image_array = np.array(image)  # Shape: (height, width, 3)
        mask_array = np.array(mask)    # Shape: (height, width)
        
        # Create an RGB image filled with the specified fill value
        fill_image = np.full_like(image_array, fill_value)  # Shape: (height, width, 3)
        
        # Apply the mask: where mask is white (>127), use fill_image; otherwise, keep original
        mask_array = mask_array[:, :, np.newaxis]  # Shape: (height, width, 1)
        result_array = np.where(mask_array > 127, fill_image, image_array)
        
        # Convert back to image and save as RGB
        result = Image.fromarray(result_array.astype(np.uint8), mode='RGB')
        result.save(output_path, quality=100)  # Use quality=100 for JPEG to maintain quality
        return True
    
    except Exception as e:
        print(f"Error processing '{input_path}' with mask '{mask_path}': {e}")
        return False

def main():
    """
    Main function to parse command-line arguments and process paired images and masks.
    """
    parser = argparse.ArgumentParser(description="Replace background regions in images using masks with a constant RGB value.")
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Directory containing input images (JPEG or PNG)"
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        required=True,
        help="Directory containing mask images (JPEG or PNG)"
    )
    parser.add_argument(
        "--fill-value",
        type=int,
        nargs=3,
        default=(5, 5, 5),
        help="RGB value to fill background (default: 5 5 5)"
    )
    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input directory '{args.input_folder}' does not exist.")
        exit()
    if not os.path.isdir(args.mask_folder):
        print(f"Error: Mask directory '{args.mask_folder}' does not exist.")
        exit()

    # Find paired files
    paired_files = find_paired_files(args.input_folder, args.mask_folder)
    if not paired_files:
        print(f"No paired image-mask files found in '{args.input_folder}' and '{args.mask_folder}'.")
        exit()

    print(f"Found {len(paired_files)} paired image-mask files to process.")

    # Process each pair
    processed_count = 0
    for input_file, mask_file in paired_files:
        input_path = os.path.join(args.input_folder, input_file)
        mask_path = os.path.join(args.mask_folder, mask_file)
        output_path = input_path  # Overwrite the original input file
        
        print(f"Processing: '{input_file}' with mask '{mask_file}'")
        if process_image_pair(input_path, mask_path, output_path, tuple(args.fill_value)):
            processed_count += 1
            print(f"Overwritten: '{output_path}'")
        else:
            print(f"Failed to process: '{input_file}'")

    print(f"\nProcessing complete. Successfully processed {processed_count}/{len(paired_files)} files.")

if __name__ == "__main__":
    main()
