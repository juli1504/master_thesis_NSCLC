"""
Dataset Splitting Script.

This script adds a 'dataset_split' column to the manifest, stratifying 
by histology to ensure an equal distribution of cancer subtypes across 
the training, validation, and test sets. It explicitly filters out ambiguous 
"NOS" cases and rare subtypes that lack enough samples to be split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

def main():
    print("Creating stratified data split...")
    
    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    
    # 1. Filter for patients that passed QC and have patches
    valid_df = df[df['patch_extracted'] == True].copy()
    
    print("\n--- Original Histology Distribution ---")
    print(valid_df['histology'].value_counts(dropna=False))
    
    # --- 2. EXPLICIT NOS FILTER ---
    # Remove any rows where the histology string contains "NOS" (case-insensitive)
    initial_count = len(valid_df)
    valid_df = valid_df[~valid_df['histology'].str.contains('NOS', case=False, na=False)]
    nos_removed = initial_count - len(valid_df)
    if nos_removed > 0:
        print(f"\n✅ Removed {nos_removed} 'NOS' (Not Otherwise Specified) cases to ensure clean Ground Truth.")

    # --- 3. RARE CLASS SAFETY FILTER ---
    # We still need at least 4 samples of any remaining class so they can survive the split
    class_counts = valid_df['histology'].value_counts()
    valid_classes = class_counts[class_counts >= 4].index
    
    split_df = valid_df[valid_df['histology'].isin(valid_classes)].copy()
    
    excluded_count = len(valid_df) - len(split_df)
    if excluded_count > 0:
        print(f"⚠️ Excluded {excluded_count} additional patients because their cancer subtype is too rare to divide.")
    
    # --- 4. PERFORM THE SPLITS ---
    # Split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        split_df, 
        test_size=0.30, 
        stratify=split_df['histology'], 
        random_state=42 # Fixed seed for reproducibility
    )
    
    # Split Temp into Validation (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['histology'], 
        random_state=42
    )
    
    # --- 5. UPDATE MANIFEST ---
    df['dataset_split'] = 'Excluded'
    df.loc[train_df.index, 'dataset_split'] = 'Train'
    df.loc[val_df.index, 'dataset_split'] = 'Validation'
    df.loc[test_df.index, 'dataset_split'] = 'Test'
    
    # Save the updated manifest
    df.to_csv(FILE_MANIFEST, index=False, sep=';', decimal=',')
    
    print("\nSplit complete. Final Distribution in Manifest:")
    print(df['dataset_split'].value_counts())

if __name__ == "__main__":
    main()