#!/usr/bin/env python3

import os
import openslide
import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def calculate_otsu_threshold(svs_file_path, ori_reso=0.25, tar_reso=5):
    """
    Calculate the Otsu threshold for a given .svs whole-slide image.

    Args:
        svs_file_path (str): Path to the .svs file.
        ori_reso (float): Original resolution in µm/pixel (default: 0.25).
        tar_reso (float): Target resolution for thumbnail in µm/pixel (default: 5).

    Returns:
        tuple: (filename, threshold) where filename is the base name of the .svs file,
               and threshold is the calculated Otsu threshold or None if an error occurs.
    """
    try:
        # Load the .svs file
        slide = openslide.OpenSlide(svs_file_path)
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
        # Print error and return filename with None threshold
        print(f"Error opening {svs_file_path}: {str(e)}")
        return os.path.basename(svs_file_path), None
    
    # Calculate the ratio for thumbnail size
    ratio = tar_reso / ori_reso
    
    # Get a thumbnail image of the slide
    thumbnail = slide.get_thumbnail((int(slide.dimensions[0] // ratio), int(slide.dimensions[1] // ratio)))
    
    # Convert the thumbnail to grayscale
    img_gray = np.array(thumbnail.convert('L'))  # Convert to grayscale

    # Calculate Otsu's threshold
    otsu_threshold = threshold_otsu(img_gray)

    return os.path.basename(svs_file_path), otsu_threshold - 2  # Return filename and adjusted threshold

def main(top_directory_path, output_csv_path):
    """
    Process all .svs files in a directory to calculate Otsu thresholds and save to a CSV file.

    Args:
        top_directory_path (str): Path to the top-level directory containing .svs files.
        output_csv_path (str): Path to save the output CSV file.
    """
    # Validate input directory
    if not os.path.isdir(top_directory_path):
        print(f"Error: {top_directory_path} is not a valid directory.")
        return

    # Prepare a list to hold data for the CSV
    data = []

    # Collect all .svs files
    svs_files = []
    for root, _, files in os.walk(top_directory_path):
        for filename in files:
            if filename.endswith('.svs'):
                svs_file_path = os.path.join(root, filename)
                svs_files.append(svs_file_path)

    if not svs_files:
        print(f"No .svs files found in {top_directory_path}.")
        return

    print(f"Found {len(svs_files)} .svs files to process.")

    # Process the files in parallel using 16 cores
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(calculate_otsu_threshold, svs_file): svs_file for svs_file in svs_files}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                filename, otsu_threshold = result
                # Append filename and threshold (or empty string if None)
                data.append({
                    'File Name': os.path.splitext(filename)[0],
                    'Otsu Threshold': otsu_threshold if otsu_threshold is not None else ''
                })

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Save to CSV
    try:
        df.to_csv(output_csv_path, index=False)
        print(f'Data saved to {output_csv_path}')
    except Exception as e:
        print(f"Error saving CSV to {output_csv_path}: {str(e)}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Calculate Otsu thresholds for .svs files and save to CSV.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the top-level directory containing .svs files"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save the output CSV file"
    )
    args = parser.parse_args()

    # Run the main function with provided arguments
    main(args.input_dir, args.output_csv)
