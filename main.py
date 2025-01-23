import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  # (If unused, you may remove this import)

# Imported functions from functions.py
from source.functions import (
    detect_edges,
    hough_transform,
    lines_to_merged,
    draw_merged_lines,
    visual_corner,
    measure_mixing_multi_lines
)

# Global parameters for easy adjustment
N_POINTS = 5       # Number of divisions for each segment (b->c, e->d)
NUM_SAMPLES = 50   # Number of pixel samples per line segment

def main():
    """
    Main entry point performing:
      1) Edge detection and average image computation
      2) Hough transform for line detection
      3) Line merging
      4) Corner visualization
      5) Mixing measurement
    """
    try:
        # Define paths
        input_dir = "./data"
        output_path = "./results"
        video = "video.mp4"

        input_path = os.path.join(input_dir, video)
        os.makedirs(output_path, exist_ok=True)

        # Step 1: Detect edges and compute an average image
        edges, avg_img = detect_edges(input_path, output_path)
        if edges is None or avg_img is None:
            print("[Error] Edge detection failed. Exiting.")
            return

        # Step 2: Apply Hough Transform to detect lines
        lines, _ = hough_transform(edges, avg_img, output_path)
        if lines is None:
            print("[Info] No lines detected. Exiting.")
            return

        # Step 3: Merge detected lines
        merged_lines = lines_to_merged(lines)
        if not merged_lines:
            print("[Info] No lines to merge. Exiting.")
            return

        # Step 4: Draw merged lines on the average image
        draw_merged_lines(avg_img, merged_lines, output_path)

        # Step 5: Visualize and extract corners (b, c, d, e)
        corners = visual_corner(merged_lines, avg_img, output_path)
        if corners is None:
            print("[Info] Could not identify corners. Exiting.")
            return

        b, c, d, e = corners

        # Step 6: Measure mixing along the specified lines
        measure_mixing_multi_lines(
            input_video=input_path,
            b=b, c=c,
            e=e, d=d,
            output_path=output_path,
            n_points=N_POINTS,
            num_samples=NUM_SAMPLES
        )

    except Exception as e:
        print(f"[Error in main] {e}")


if __name__ == "__main__":
    main()