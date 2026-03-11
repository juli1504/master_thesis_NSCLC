"""
CT Window Width Sweep Experiment Script.

This script acts as the second part of the visual radiological walkthrough. 
While the previous script swept the Window Center (Level), this script fixes 
the center at -600 HU (the standard lung density) and systematically sweeps 
the Window Width (Contrast) from 100 HU to 2400 HU. This generates a 24-step 
matrix to empirically determine the optimal contrast setting for visualizing 
lung tumors and parenchyma in the deep learning pipeline.
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import math

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

def load_middle_slice(path):
    """
    Loads only the middle slice from a DICOM folder for testing purposes.

    The function reads all '.dcm' files in the directory, sorts them anatomically 
    along the Z-axis using 'ImagePositionPatient', and returns the median slice.

    Args:
        path (Path or str): The directory path containing the DICOM files.

    Returns:
        pydicom.dataset.FileDataset or None: The middle DICOM slice object, 
        or None if the directory is empty.
    """
    files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    if not files: return None
    
    # Sorting is important for consistent results
    slices = [pydicom.dcmread(path / f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    return slices[len(slices) // 2]

def get_pixels_hu(slice_item):
    """
    Converts a single DICOM slice into Hounsfield Units (HU).

    This function applies the linear transformation defined in the DICOM header 
    (RescaleIntercept and RescaleSlope) to convert scanner pixel values into 
    standardized radiodensity metrics (Hounsfield Units). 

    Args:
        slice_item (pydicom.dataset.FileDataset): The single DICOM object.

    Returns:
        numpy.ndarray: A 2D array containing the calculated HU values.
    """
    image = slice_item.pixel_array.astype(np.int16)
    image[image == -2000] = -1000
    
    intercept = slice_item.RescaleIntercept
    slope = slice_item.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    return image

def apply_window(image, center, width):
    """
    Applies radiological CT windowing to restrict the visible HU range.

    Args:
        image (numpy.ndarray): The input 2D image array in Hounsfield Units.
        center (int): The center value (Level) of the window in HU.
        width (int): The width of the window in HU.

    Returns:
        numpy.ndarray: The windowed (clipped) image array.
    """
    img_min = center - width // 2
    img_max = center + width // 2
    return np.clip(image, img_min, img_max)

def main():
    """
    Executes the fine-grained window width experiment and generates a visualization grid.

    The function performs the following steps:
    1. Loads the exact DICOM path for a sample patient (AMC-003).
    2. Extracts the middle slice and converts it to Hounsfield Units.
    3. Fixes the Window Center at -600 HU (lung standard).
    4. Sweeps the Window Width from 100 to 2400 in increments of 100.
    5. Plots each resulting windowed image into a 4x6 matplotlib grid.
    6. Saves the final grid as a 'poster' image.

    Returns:
        None. The resulting grid is saved to the 'results/figures' directory.
    """
    print("### STEP 5: WIDTH EXPERIMENT (Fine-Grained 24-Step Matrix) ###")
    
    # 1. Load data (AMC-003)
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['Subject ID'] == 'AMC-003'].iloc[0]
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    full_path = DIR_DICOM / clean_path
    
    mid_slice = load_middle_slice(full_path)
    img_hu = get_pixels_hu(mid_slice)
    
    # 2. Experiment Configuration
    fixed_center = -600  # We fix the brightness to the lung standard
    
    # We create 24 steps from 100 to 2400 (step size 100)
    # This perfectly covers the range from "way too hard" to "way too soft"
    widths = list(range(100, 2500, 100)) 
    
    num_plots = len(widths)
    cols = 4  # 4 columns
    rows = math.ceil(num_plots / cols) # Should result in 6
    
    print(f"Creating {num_plots} images (Width 100 to 2400 in steps of 100)...")
    
    # Large portrait format for the 24 images
    plt.figure(figsize=(16, 24))
    
    for i, width in enumerate(widths):
        ax = plt.subplot(rows, cols, i + 1)
        
        img_windowed = apply_window(img_hu, center=fixed_center, width=width)
        
        ax.imshow(img_windowed, cmap='gray')
        
        # Format title
        title_color = 'black'
        bg_color = 'white'
        weight = 'normal'

        ax.set_title(f"Width: {width} HU", fontsize=12, color=title_color, weight=weight)
        ax.axis('off')
            
    plt.suptitle(f"Contrast Variation (Fixed Center: {fixed_center} HU)", fontsize=20, y=0.99)
    plt.tight_layout()
    
    save_path = DIR_FIGURES / "width_walkthrough_24steps.png"
    plt.savefig(save_path, dpi=150)
    print(f">> Poster saved: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()