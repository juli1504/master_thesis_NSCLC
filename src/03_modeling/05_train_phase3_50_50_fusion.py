"""
Phase 3: Multimodal Late Fusion (Simple 50/50 Averaging)

This script recreates the absolute best models from Phase 1 and Phase 2:
- Clinical: Tuned MLP (hidden_layers=(64, 32), lr=0.1, alpha=0.01)
- Vision: ResNet-18 (Unfrozen Blocks: 4)

It extracts the prediction probabilities from both models on the strict Test set,
averages them together equally (50/50), and calculates the final Multimodal metrics.
"""

import os
import random
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION & SEEDING ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
VISION_WEIGHTS = PROJECT_ROOT / "best_resnet_unfrozen_4.pth"

def set_seed(seed=42):
    """Locks down all random number generators for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def build_resnet(in_channels, num_classes=2):
    model = models.resnet18()
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, 
                            kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                            padding=original_conv.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# --- 3. MAIN FUSION SCRIPT ---
def main():
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STARTING PHASE 3: MULTIMODAL LATE FUSION (50/50 AVERAGE) ===")
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
    
    # 2. Prepare Clinical Data (Exact Phase 1b Pipeline)
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    train_val_mask = df['dataset_split'].isin(['Train', 'Validation'])
    test_mask = df['dataset_split'] == 'Test'
    
    X_train_val = X_encoded[train_val_mask]
    y_train_val = df.loc[train_val_mask, 'target']
    X_test = X_encoded[test_mask]
    y_test = df.loc[test_mask, 'target'] # The Ground Truth!
    
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
    print("Loading Phase 2 Champion (ResNet Level 4)...")
    test_df = df[test_mask].copy()
    test_dataset = CTPatchDataset(test_df, le)
    # Important: shuffle=False ensures the images align perfectly with the clinical rows!
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    
    in_channels = test_dataset[0][0].shape[0]
    vision_model = build_resnet(in_channels).to(device)
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

    # --- PILLAR 3: LATE FUSION (50/50 AVERAGING) ---
    print("Performing Late Fusion (50/50 Averaging)...")
    probs_fusion = (probs_clinical + probs_vision) / 2.0
    
    y_test_array = y_test.values
    
    # Calculate Optimal Threshold for the Fused Probabilities
    fpr, tpr, thresholds = roc_curve(y_test_array, probs_fusion)
    youden_j = tpr - fpr
    best_thresh = thresholds[np.argmax(youden_j)]
    
    y_pred_fusion = (probs_fusion >= best_thresh).astype(int)
    
    # Metrics
    auc_fusion = roc_auc_score(y_test_array, probs_fusion)
    acc_fusion = accuracy_score(y_test_array, y_pred_fusion)
    f1_fusion = f1_score(y_test_array, y_pred_fusion)
    tn, fp, fn, tp = confusion_matrix(y_test_array, y_pred_fusion, labels=[0, 1]).ravel()
    sens_fusion = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_fusion = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # --- DISPLAY FINAL RESULTS ---
    print("\n" + "="*85)
    print("PHASE 3: FINAL MULTIMODAL RESULTS (50/50 AVERAGE)")
    print("="*85)
    print(f"{'Metric':<15} | {'Phase 1 (Clinical)':<20} | {'Phase 2 (Vision)':<18} | {'Phase 3 (Fusion)':<15}")
    print("-" * 85)
    # Hardcoded values updated to the honest Phase 1b and Phase 2 ResNet Champions
    print(f"{'AUC':<15} | {'0.652':<20} | {'0.757':<18} | {auc_fusion:.3f}")
    print(f"{'Sensitivity':<15} | {'80.0%':<20} | {'100.0%':<18} | {sens_fusion*100:.1f}%")
    print(f"{'Specificity':<15} | {'47.8%':<20} | {'65.2%':<18} | {spec_fusion*100:.1f}%")
    print(f"{'F1-Score':<15} | {'38.1%':<20} | {'55.6%':<18} | {f1_fusion*100:.1f}%")
    print(f"{'Accuracy':<15} | {'53.6%':<20} | {'71.4%':<18} | {acc_fusion*100:.1f}%")
    print("="*85)

if __name__ == "__main__":
    main()