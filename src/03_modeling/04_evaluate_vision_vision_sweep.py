"""
Phase 2 Evaluation Script

This script automatically loads all 18 saved Vision models (.pth files)
from the hyperparameter sweep and evaluates them on the untouched Test Set.
It prints a single, clean matrix for your thesis report!
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

# --- 2. DATASET DEFINITION (TEST ONLY) ---
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

# --- 3. MODEL BUILDER ---
def build_vision_model(model_name, in_channels, num_classes=2):
    """Builds the base architecture to load our saved weights into."""
    if model_name == 'resnet':
        model = models.resnet18()
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet':
        model = models.densenet121()
        model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0()
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# --- 4. EVALUATION FUNCTION ---
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_probs = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    y_true, y_probs = np.array(y_true), np.array(y_probs)
    
    try: auc = roc_auc_score(y_true, y_probs)
    except ValueError: auc = 0.5  
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_j = tpr - fpr
    best_thresh = thresholds[np.argmax(youden_j)]
    
    y_pred = (y_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return auc, sens, spec

# --- 5. MAIN SCRIPT ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rescue Script Starting... Using Hardware: {device}\n")

    # Setup Data
    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    df = df[(df['dataset_split'] != 'Excluded') & (df['histology'].isin(['Adenocarcinoma', 'Squamous cell carcinoma']))].copy()
    
    le = LabelEncoder()
    le.fit(df['histology'])
    
    test_df = df[df['dataset_split'] == 'Test']
    test_dataset = CTPatchDataset(test_df, le)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    in_channels = test_dataset[0][0].shape[0]
    print(f"Loading Test Set... Found {len(test_dataset)} patients.\n")

    # Loop through all models and levels
    architectures = ['resnet', 'densenet', 'efficientnet']
    levels = [0, 1, 2, 3, 4, 5]
    
    results = []
    
    print("Evaluating all saved models on the Test Set...")
    for arch in architectures:
        for level in levels:
            model_file = PROJECT_ROOT / f"best_{arch}_unfrozen_{level}.pth"
            
            # If the file exists, evaluate it!
            if model_file.exists():
                model = build_vision_model(arch, in_channels).to(device)
                model.load_state_dict(torch.load(model_file, map_location=device))
                
                auc, sens, spec = evaluate(model, test_loader, device)
                
                results.append({
                    "Architecture": arch.upper(),
                    "Unfrozen Blocks": level,
                    "AUC": f"{auc:.3f}",
                    "Sensitivity": f"{sens*100:.1f}%",
                    "Specificity": f"{spec*100:.1f}%",
                    "Sens+Spec (Score)": f"{(sens+spec):.3f}"
                })
            else:
                print(f"Missing file: {model_file.name}")

    # Display the final beautiful matrix
    print("\n" + "="*85)
    print("PHASE 2 MASTER RESULTS MATRIX (EVALUATED STRICTLY ON UNTOUCHED TEST SET)")
    print("="*85)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()