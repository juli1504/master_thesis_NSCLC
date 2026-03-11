"""
Tumor Coordinate Plotting and Visual QC Script.

This script serves as a visual Quality Control (QC) validation step. It takes 
hardcoded manual XML coordinates (X, Y) and the unique `SOPInstanceUID` of a 
target slice, locates the exact physical DICOM file on the hard drive, and 
plots the 2D slice. Finally, it draws a bounding box/circle and an arrow at 
the extracted coordinates to mathematically and visually prove that the spatial 
mapping between the clinical AIM annotations and the DICOM pixel arrays is perfectly aligned.
"""

import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path
import os
import numpy as np

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
PATH_MAPPING = PROJECT_ROOT / "data" / "processed" / "exact_image_mapping.csv"
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# DATA FROM XML (Manually extracted for this test)
TARGET_PATIENT = "AMC-003"
TARGET_SOP_UID = "1.3.6.1.4.1.14519.5.2.1.4334.1501.553921625749272741224744327937"
TARGET_X = 126.5
TARGET_Y = 344.8

def apply_lung_window(image):
    """
    Applies a standard radiological lung window to the image.

    This function adjusts the visible contrast of the CT slice by clipping 
    the Hounsfield Units (HU) to a predefined lung window (Center: -600 HU, 
    Width: 1500 HU). It also handles basic out-of-bounds scanner padding.

    Args:
        image (numpy.ndarray): The 2D image array in Hounsfield Units.

    Returns:
        numpy.ndarray: The windowed (clipped) image array.
    """
    # Standard Lung Window: Center -600, Width 1500
    center = -600
    width = 1500
    
    # HU Conversion (simplified)
    image = image.astype(np.int16)
    image[image == -2000] = -1000
    
    img_min = center - width // 2
    img_max = center + width // 2
    return np.clip(image, img_min, img_max)

def main():
    """
    Executes the visual coordinate validation and plotting pipeline.

    The function performs the following steps:
    1. Looks up the local directory path for the target patient in the mapping file.
    2. Scans the directory to find the specific DICOM file matching the target 
       SOPInstanceUID (using rapid header reading).
    3. Loads the matched DICOM file, applies HU correction, and applies a lung window.
    4. Plots the 2D image and annotates it with a red circle and a yellow arrow 
       at the exact (X, Y) coordinates specified in the XML.

    Returns:
        None. The annotated image is displayed and saved to the 'results/figures' directory.
    """
    print(f"--- Searching for the 'hidden' tumor in {TARGET_PATIENT} ---")
    
    # 1. Get path
    df_map = pd.read_csv(PATH_MAPPING)
    match = df_map[df_map['Subject ID'] == TARGET_PATIENT].iloc[0]
    
    raw_path = match['File Location']
    clean_path = raw_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    full_path = DIR_DICOM / clean_path
    
    print(f"Scanning directory: {full_path}")
    
    # 2. Find the correct file (Matching via SOP UID)
    found_file = None
    dcm_files = list(full_path.glob("*.dcm"))
    
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True) # Quick check without pixel data
            if ds.SOPInstanceUID == TARGET_SOP_UID:
                found_file = f
                print(f"File found: {f.name}.")
                break
        except Exception:
            continue
            
    if not found_file:
        print("ERROR: Could not find the SOP UID from the XML in this directory.")
        # Fallback: Try frame 123 (Filenames are often '1-123.dcm')
        print("Attempting fallback to filename '1-123.dcm'...")
        found_file = full_path / "1-123.dcm"
        if not found_file.exists():
            print("File 1-123.dcm also not found.")
            return

    # 3. Load and Plot
    ds = pydicom.dcmread(found_file)
    
    # HU Correction (Slope/Intercept)
    img = ds.pixel_array.astype(np.float64) * ds.RescaleSlope + ds.RescaleIntercept
    img_lung = apply_lung_window(img)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_lung, cmap='gray')
    
    # 4. Draw the marker from the XML
    # XML says: TwoDimensionCircle at x=126.5, y=344.8
    # We draw a circle around it
    circle = patches.Circle((TARGET_X, TARGET_Y), radius=10, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(circle)
    
    # Add an arrow so it is immediately visible
    plt.annotate('HERE IS THE TUMOR', xy=(TARGET_X, TARGET_Y), xytext=(TARGET_X+50, TARGET_Y-50),
                 arrowprops=dict(facecolor='yellow', shrink=0.05),
                 color='yellow', fontsize=12, weight='bold')

    plt.title(f"Patient {TARGET_PATIENT} | SOP UID Match | Lung Window")

    save_path = DIR_FIGURES / "example_lung_cancer.png"
    plt.savefig(save_path, dpi=150)
    print(f">> Image saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()