"""
2.5D Patch Extraction Script for Deep Learning.

This script isolates the tumor and extracts machine learning-ready 3D tensors 
from the full DICOM volumes. It uses a 2.5D approach, extracting a 128x128 pixel 
window in the X/Y plane and extending 3 slices above and below the target Z-axis 
slice (resulting in a 128x128x7 tensor). The script handles anatomical Z-axis 
sorting, Hounsfield Unit conversion, and edge-case padding (using -1000 HU / Air) 
for tumors located near the lung boundaries. Extracted patches are saved as numpy `.npy` arrays.
"""

import pandas as pd
import pydicom
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION (Defining the professor's specifications here) ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIR_DICOM = PROJECT_ROOT / "data" / "raw" / "dicom"
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

# PATCH PARAMETERS (Documented for protocol & manifest)
PATCH_SIZE_XY = 128
PATCH_SLICES_Z_PLUS_MINUS = 3  # +/- 3 Slices -> Total of 7 Slices (2.5D)

DIR_PATCHES = PROJECT_ROOT / "data" / "processed" / "patches_2_5D"
DIR_PATCHES.mkdir(parents=True, exist_ok=True)

def transform_to_hu(dicom_ds):
    """
    Converts raw DICOM pixel values into Hounsfield Units (HU).

    Args:
        dicom_ds (pydicom.dataset.FileDataset): The loaded DICOM file object.

    Returns:
        numpy.ndarray: The image array converted to Hounsfield Units (as float64).
    """
    image = dicom_ds.pixel_array.astype(np.float64)
    intercept = dicom_ds.RescaleIntercept if 'RescaleIntercept' in dicom_ds else -1024.0
    slope = dicom_ds.RescaleSlope if 'RescaleSlope' in dicom_ds else 1.0
    return (image * slope) + intercept

def get_sorted_dicom_series(patient_dir, target_series_uid):
    """
    Searches for all DICOMs in a series and sorts them anatomically by Z-coordinate.

    Args:
        patient_dir (Path): Directory containing the patient's DICOM files.
        target_series_uid (str): The SeriesInstanceUID to extract.

    Returns:
        list of tuples: A list sorted descending by Z-coordinate. Each tuple 
        contains (z_position, file_path, SOPInstanceUID).
    """
    slices = []
    for root_dir, dirs, files in os.walk(patient_dir):
        dicom_files = [f for f in files if f.endswith('.dcm')]
        for f in dicom_files:
            dcm_path = Path(root_dir) / f
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                if str(ds.SeriesInstanceUID).strip().replace('\x00', '') == target_series_uid:
                    # Use ImagePositionPatient[2] (Z-coordinate) for true anatomical sorting
                    z_pos = float(ds.ImagePositionPatient[2]) if 'ImagePositionPatient' in ds else float(ds.InstanceNumber)
                    slices.append((z_pos, dcm_path, str(ds.SOPInstanceUID).strip().replace('\x00', '')))
            except Exception:
                continue
    
    # Sort by Z-coordinate (descending, i.e., head to toe, standard for CTs)
    slices.sort(key=lambda x: x[0], reverse=True)
    return slices

def extract_2_5d_patch(slices_info, target_sop, x_center, y_center):
    """
    Extracts a 128x128x7 patch around the target coordinates.

    This function handles the extraction of the multi-slice tensor. It automatically 
    applies Hounsfield Unit conversion and pads the image with -1000 HU (Air) 
    if the 128x128 bounding box extends beyond the edges of the original DICOM array.

    Args:
        slices_info (list): Sorted list of tuples from `get_sorted_dicom_series`.
        target_sop (str): The SOPInstanceUID of the center slice containing the tumor.
        x_center (int): The X pixel coordinate of the tumor center.
        y_center (int): The Y pixel coordinate of the tumor center.

    Returns:
        tuple: (numpy.ndarray, str). The 3D patch array of shape (7, 128, 128) 
        and a status message ("Success" or an error string). Returns (None, error) on failure.
    """
    # Find the index of the target image (where the tumor is marked)
    target_idx = -1
    for i, (_, _, sop) in enumerate(slices_info):
        if sop == target_sop:
            target_idx = i
            break
            
    if target_idx == -1:
        return None, "Target SOP not found in series"

    # Determine start and end indices for the Z-axis (Z-Clipping / Padding)
    z_start = target_idx - PATCH_SLICES_Z_PLUS_MINUS
    z_end = target_idx + PATCH_SLICES_Z_PLUS_MINUS
    
    patch_volume = []
    
    half_size = PATCH_SIZE_XY // 2
    
    # Iterate over the 7 required slices
    for z in range(z_start, z_end + 1):
        if 0 <= z < len(slices_info):
            # Normal case: Slice exists
            ds = pydicom.dcmread(slices_info[z][1])
            img_hu = transform_to_hu(ds)
        else:
            # Edge Case (Edge of the CT scan): We duplicate the outermost existing image
            valid_z = max(0, min(z, len(slices_info) - 1))
            ds = pydicom.dcmread(slices_info[valid_z][1])
            img_hu = transform_to_hu(ds)

        # X/Y Cropping with edge padding (if tumor is at the very edge of the lung)
        padded_img = np.pad(img_hu, pad_width=half_size, mode='constant', constant_values=-1000) # -1000 is Air
        
        # IMPORTANT: Padding shifts the center by half_size
        y_start = y_center
        y_end = y_center + PATCH_SIZE_XY
        x_start = x_center
        x_end = x_center + PATCH_SIZE_XY
        
        patch_2d = padded_img[y_start:y_end, x_start:x_end]
        patch_volume.append(patch_2d)

    # Assemble into a 3D Array: Shape (7, 128, 128) -> (Z, Y, X)
    patch_3d = np.stack(patch_volume, axis=0)
    return patch_3d, "Success"

def main():
    """
    Executes the batched 2.5D patch extraction pipeline.

    The function performs the following operations:
    1. Loads the manifest and filters for successfully mapped patients.
    2. Iterates through the cohort, sorting their DICOM slices anatomically.
    3. Extracts the 128x128x7 tensor around the mapped tumor coordinates.
    4. Saves the extracted tensor as a numpy `.npy` file.
    5. Updates the central manifest with patch extraction status and file paths.
    """
    print("Starting 2.5D Patch Extraction...")
    
    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    mask = (df['coordinate_mapped_successfully'] == True)
    patients_to_process = df[mask].copy()
    
    print(f"Processing {len(patients_to_process)} validated patients...")
    
    # New columns in the manifest for documentation
    df['patch_size_xy'] = PATCH_SIZE_XY
    df['patch_slices_z'] = PATCH_SLICES_Z_PLUS_MINUS * 2 + 1
    df['patch_extracted'] = False
    df['patch_file_path'] = None
    
    for idx, row in tqdm(patients_to_process.iterrows(), total=len(patients_to_process)):
        pid = row['subject_id']
        t_series = str(row['chosen_series_uid']).strip()
        t_sop = str(row['sop_instance_uid']).strip()
        x_pixel = int(row['x_pixel'])
        y_pixel = int(row['y_pixel'])
        
        patient_dir = DIR_DICOM / pid
        if not patient_dir.exists():
            patient_dir = DIR_DICOM / "NSCLC Radiogenomics" / pid
            
        # 1. Load all DICOMs of the series and sort them
        slices_info = get_sorted_dicom_series(patient_dir, t_series)
        
        if not slices_info:
            continue
            
        # 2. Cut out 2.5D Patch
        patch_array, status = extract_2_5d_patch(slices_info, t_sop, x_pixel, y_pixel)
        
        if status == "Success":
            # 3. Save patch (.npy is the standard format for ML Numpy Arrays)
            filename = f"{pid}_patch_128_2.5D.npy"
            filepath = DIR_PATCHES / filename
            np.save(filepath, patch_array)
            
            # 4. Document in the manifest
            df.at[idx, 'patch_extracted'] = True
            df.at[idx, 'patch_file_path'] = str(filepath.relative_to(PROJECT_ROOT))
            
    # Update Manifest
    df.to_csv(FILE_MANIFEST, index=False, sep=';', decimal=',')
    
    print("\n" + "="*50)
    print("PATCH EXTRACTION COMPLETE")
    print("="*50)
    print(f"Patches saved in: {DIR_PATCHES}")
    print(f"Manifest updated: {FILE_MANIFEST}")
    
if __name__ == "__main__":
    main()