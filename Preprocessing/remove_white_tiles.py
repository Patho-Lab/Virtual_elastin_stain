#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
import argparse

def is_mostly_white(tile, expected_threshold=0.8, white_threshold=220):
    """
    Check if an image tile is mostly white based on a threshold.

    Args:
        tile (numpy.ndarray): Image tile loaded with OpenCV.
        expected_threshold (float): Minimum white pixel ratio to consider the tile "mostly white" (default: 0.8).
        white_threshold (int): Grayscale intensity threshold for white pixels (default: 220).

    Returns:
        bool: True if the white pixel ratio exceeds expected_threshold, False otherwise.
    """
    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    white_ratio = np.sum(gray_tile >= white_threshold) / gray_tile.size
    return white_ratio > expected_threshold

def process_tile(tile_path, white_threshold):
    """
    Process a single image tile and remove it if it is mostly white.

    Args:
        tile_path (str): Path to the image tile.
        white_threshold (float): Grayscale intensity threshold for white pixels.
    """
    tile = cv2.imread(tile_path)
    if tile is None:
        print(f"Warning: Unable to read tile at {tile_path}")
        return
    if is_mostly_white(tile, white_threshold=white_threshold):
        try:
            os.remove(tile_path)
            print(f"Removed mostly white tile: {tile_path}")
        except Exception as e:
            print(f"Error removing tile {tile_path}: {e}")

def load_thresholds(csv_file_path):
    """
    Load white thresholds from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file with columns for filenames and thresholds.

    Returns:
        dict: Dictionary mapping directory names to their respective thresholds.
              Returns empty dict if loading fails.
    """
    try:
        df = pd.read_csv(csv_file_path)
        return dict(zip(df.iloc[:, 0], df.iloc[:, 1].astype(float)))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}

def remove_white_tiles(root_directory, thresholds):
    """
    Remove mostly white image tiles from a directory using thresholds.

    Args:
        root_directory (str): Root directory containing image tiles.
        thresholds (dict): Dictionary mapping directory names to white thresholds.
    """
    if not os.path.isdir(root_directory):
        print(f"Error: {root_directory} is not a valid directory.")
        return

    print(f"Starting the removal process in {root_directory}")
    
    # Create a pool of worker processes
    with Pool(processes=16) as pool:
        for root, _, files in os.walk(root_directory, followlinks=True):
            # Get the threshold for the current directory, default to 220 if not found
            white_threshold = thresholds.get(os.path.basename(root), 220)
            jpg_files = [os.path.join(root, file) for file in files if file.lower().endswith((".jpg", ".png"))]
            
            # Process each image file in parallel using the pool
            pool.starmap(process_tile, [(tile_path, white_threshold) for tile_path in jpg_files])

    print("Removal process completed.")

def main(tiles_directory, csv_file_path):
    """
    Main function to load thresholds and remove white tiles.

    Args:
        tiles_directory (str): Path to the directory containing image tiles.
        csv_file_path (str): Path to the CSV file with thresholds.
    """
    thresholds = load_thresholds(csv_file_path)
    if thresholds:
        remove_white_tiles(tiles_directory, thresholds)
    else:
        print("No thresholds available. Exiting.")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Remove mostly white image tiles based on thresholds from a CSV file.")
    parser.add_argument(
        "--tiles-dir",
        type=str,
        required=True,
        help="Path to the directory containing image tiles (JPEG or PNG)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to the CSV file containing thresholds"
    )
    args = parser.parse_args()

    # Run the main function with provided arguments
    main(args.tiles_dir, args.csv_file)
