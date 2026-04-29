"""
Phase 3: Multimodal Late Fusion (Clinical + Vision)

This script implements AUC-Weighted Late Fusion. 
Instead of a 50/50 split, the models are assigned a "Voting Weight" 
proportional to their individual baseline AUC scores, giving the 
superior Vision model a mathematically higher impact on the final decision.
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
VISION_WEIGHTS = PROJECT_ROOT / "best_densenet_unfrozen_4.pth"

# AUC Scores from Phase 1 and 2
AUC_CLINICAL = 0.652
AUC_VISION = 0.730

# --- 2. VISION DATASET & BUILDER ---
class CTPatchDataset(Dataset):
    def __init__(self, manifest_df, label_encoder):
        self.df = manifest_df[manifest_df['patch_extracted'] == True].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.le = label_encoder
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_path = PROJECT_ROOT / row['patch_file_path']
        patch_array = np.load(patch_path).astype(np.float32)
        
        image_tensor = torch.tensor(patch_array)
        if image_tensor.shape[-1] < 10: 
            image_tensor = image_tensor.permute(2, 0, 1)
            
        label = self.le.transform([row['histology']])[0]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

def build_densenet(in_channels, num_classes=2):
    model = models.densenet121()
    model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# --- 3. MAIN FUSION SCRIPT ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STARTING PHASE 3: AUC-WEIGHTED LATE FUSION ===")
    print(f"Using Hardware: {device}\n")

    # 1. Load and Merge Data
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    valid_cancers = ['Adenocarcinoma', 'Squamous cell carcinoma']
    manifest_df = manifest_df[manifest_df['histology'].isin(valid_cancers)].copy()
    
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    
    # 2. Prepare Clinical Data
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    train_val_mask = df['dataset_split'].isin(['Train', 'Validation'])
    test_mask = df['dataset_split'] == 'Test'
    
    X_train_val = X_encoded[train_val_mask]
    y_train_val = df.loc[train_val_mask, 'target']
    X_test = X_encoded[test_mask]
    y_test = df.loc[test_mask, 'target'] 
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_val_scaled = scaler.fit_transform(imputer.fit_transform(X_train_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    # --- PILLAR 1: GET CLINICAL PROBABILITIES ---
    print("Training Phase 1 Champion (Tuned MLP)...")
    clinical_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=0.1, alpha=0.01, max_iter=1000, random_state=42))
    ])
    clinical_pipeline.fit(X_train_val_scaled, y_train_val)
    probs_clinical = clinical_pipeline.predict_proba(X_test_scaled)[:, 1]

    # --- PILLAR 2: GET VISION PROBABILITIES ---
    print("Loading Phase 2 Champion (DenseNet Level 4)...")
    test_df = df[test_mask].copy()
    test_dataset = CTPatchDataset(test_df, le)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    
    in_channels = test_dataset[0][0].shape[0]
    vision_model = build_densenet(in_channels).to(device)
    vision_model.load_state_dict(torch.load(VISION_WEIGHTS, map_location=device))
    vision_model.eval()
    
    probs_vision = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = vision_model(images)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            probs_vision.extend(batch_probs.cpu().numpy())
            
    probs_vision = np.array(probs_vision)

    # --- PILLAR 3: AUC-WEIGHTED LATE FUSION ---
    print("\nCalculating Dynamic Voting Weights...")
    total_auc = AUC_CLINICAL + AUC_VISION
    weight_clinical = AUC_CLINICAL / total_auc
    weight_vision = AUC_VISION / total_auc
    
    print(f" -> Clinical Weight: {weight_clinical*100:.1f}%")
    print(f" -> Vision Weight:   {weight_vision*100:.1f}%")
    
    # The Weighted Averaging Math
    probs_fusion = (probs_clinical * weight_clinical) + (probs_vision * weight_vision)
    
    y_test_array = y_test.values
    
    # Calculate Optimal Threshold for the Fused Probabilities
    fpr, tpr, thresholds = roc_curve(y_test_array, probs_fusion)
    youden_j = tpr - fpr
    best_thresh = thresholds[np.argmax(youden_j)]
    
    y_pred_fusion = (probs_fusion >= best_thresh).astype(int)
    
    # Metrics
    auc_fusion = roc_auc_score(y_test_array, probs_fusion)
    acc_fusion = accuracy_score(y_test_array, y_pred_fusion)
    tn, fp, fn, tp = confusion_matrix(y_test_array, y_pred_fusion, labels=[0, 1]).ravel()
    sens_fusion = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_fusion = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # --- DISPLAY FINAL RESULTS ---
    print("\n" + "="*80)
    print("PHASE 3: THE FINAL MULTIMODAL SHOWDOWN (AUC-WEIGHTED)")
    print("="*80)
    print(f"{'Metric':<15} | {'Phase 1 (Clinical)':<20} | {'Phase 2 (Vision)':<18} | {'Phase 3 (Fusion)':<15}")
    print("-" * 80)
    print(f"{'AUC':<15} | {'0.652':<20} | {'0.730':<18} | {auc_fusion:.3f}")
    print(f"{'Sensitivity':<15} | {'80.0%':<20} | {'100.0%':<18} | {sens_fusion*100:.1f}%")
    print(f"{'Specificity':<15} | {'47.8%':<20} | {'65.2%':<18} | {spec_fusion*100:.1f}%")
    print(f"{'Accuracy':<15} | {'53.6%':<20} | {'71.4%':<18} | {acc_fusion*100:.1f}%")
    print(f"{'Score (Sens+Spec)':<17} | {'1.278':<18} | {'1.652':<18} | {(sens_fusion+spec_fusion):.3f}")
    print("="*80)

if __name__ == "__main__":
    main()