"""
Phase 2: Vision Baselines and Fine-Tuning (2.5D CT Patches)
Supports precise unfreezing of 0 to 5 architectural blocks to control overfitting.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

import torchvision.transforms as T

# --- 2. DATASET DEFINITION ---
class CTPatchDataset(Dataset):
    def __init__(self, manifest_df, label_encoder, transform=None):
        self.df = manifest_df[manifest_df['patch_extracted'] == True].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.le = label_encoder
        self.transform = transform # Store the augmentation rules
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        patch_path = PROJECT_ROOT / row['patch_file_path']
        patch_array = np.load(patch_path).astype(np.float32)
        
        image_tensor = torch.tensor(patch_array)
        
        if image_tensor.shape[-1] < 10: 
            image_tensor = image_tensor.permute(2, 0, 1)
            
        # Apply the physical augmentations (flipping/rotating) if they exist
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        label = self.le.transform([row['histology']])[0]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

# --- 3. MODEL BUILDER (BLOCK DIAL UPGRADE) ---
def build_vision_model(model_name, unfreeze_blocks, in_channels, num_classes=2):
    """Builds the model and unfreezes a specific number of architectural blocks."""
    
    if model_name == 'resnet':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 1. Start by freezing EVERYTHING
        for param in model.parameters(): 
            param.requires_grad = False
            
        # 2. Progressively unfreeze blocks from back to front based on user input
        if unfreeze_blocks >= 1:
            for param in model.layer4.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 2:
            for param in model.layer3.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 3:
            for param in model.layer2.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 4:
            for param in model.layer1.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 5:
            # Unfreeze the initial stem (Full fine-tuning)
            for param in model.parameters(): param.requires_grad = True

        # 3. Always make the new 7-channel input layer trainable
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, 
                                kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                                padding=original_conv.padding, bias=False)
        model.conv1.weight.requires_grad = True 
        
        # 4. Modify classification head (Always trainable)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'densenet':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
        
        # DenseNet has 4 DenseBlocks
        if unfreeze_blocks >= 1:
            for param in model.features.denseblock4.parameters(): param.requires_grad = True
            for param in model.features.norm5.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 2:
            for param in model.features.transition3.parameters(): param.requires_grad = True
            for param in model.features.denseblock3.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 3:
            for param in model.features.transition2.parameters(): param.requires_grad = True
            for param in model.features.denseblock2.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 4:
            for param in model.features.transition1.parameters(): param.requires_grad = True
            for param in model.features.denseblock1.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 5:
            for param in model.parameters(): param.requires_grad = True

        original_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(in_channels, original_conv.out_channels, 
                                         kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                                         padding=original_conv.padding, bias=False)
        model.features.conv0.weight.requires_grad = True
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
            
        # EfficientNet-B0 has 8 main feature stages. We group them roughly to match 1-4 logic.
        if unfreeze_blocks >= 1:
            for param in model.features[7:].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 2:
            for param in model.features[5:7].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 3:
            for param in model.features[3:5].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 4:
            for param in model.features[1:3].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 5:
            for param in model.parameters(): param.requires_grad = True

        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, original_conv.out_channels, 
                                         kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                                         padding=original_conv.padding, bias=False)
        model.features[0][0].weight.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

# --- 4. EVALUATION FUNCTION ---
def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Class 1 (Squamous)
            
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return acc, auc, sens, spec

# --- 5. MAIN SCRIPT ---
def main():
    # Set up arguments so you can control it via terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'densenet', 'efficientnet'])
    parser.add_argument('--unfreeze_blocks', type=int, default=1, choices=[0, 1, 2, 3, 4, 5], help="0=Frozen, 1-4=Partial, 5=Full")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    print(f"=== STARTING PHASE 2 RUN ===")
    print(f"Model: {args.model.upper()} | Unfrozen Blocks: {args.unfreeze_blocks} | Epochs: {args.epochs}")

    # Set Device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Hardware: {device}\n")

    # Load Manifest
    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    
    # Filter out the 'NOS' ghost class and Excluded patients
    df = df[df['dataset_split'] != 'Excluded'].copy()
    valid_cancers = ['Adenocarcinoma', 'Squamous cell carcinoma']
    df = df[df['histology'].isin(valid_cancers)].copy()
    
    # Label Encoder for targets
    le = LabelEncoder()
    le.fit(df['histology'])
    print(f"Target Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    # Split datasets
    train_df = df[df['dataset_split'] == 'Train']
    test_df = df[df['dataset_split'] == 'Test']

    # Define the Data Augmentation pipeline (Spatial only, no color distortion for CTs)
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15)
    ])

    # Apply the transforms ONLY to the training set
    train_dataset = CTPatchDataset(train_df, le, transform=train_transforms)
    test_dataset = CTPatchDataset(test_df, le, transform=None) # Keep test data pure!

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Automatically detect how many slices (channels) your patches have
    sample_img, _ = train_dataset[0]
    in_channels = sample_img.shape[0]
    print(f"Detected 2.5D Patches with {in_channels} channels.")

    # Calculate Class Weights to fix the Imbalance
    y_train = train_dataset.df['histology'].apply(lambda x: le.transform([x])[0])
    class_wts = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_wts, dtype=torch.float32).to(device)
    print(f"Calculated Class Weights: {weights_tensor.cpu().numpy()}\n")

    # Build Model, Loss, Optimizer
    model = build_vision_model(args.model, args.unfreeze_blocks, in_channels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # ONLY pass the unfrozen parameters to the optimizer!
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # Define the save name ONCE before the loop starts
    save_name = f"best_{args.model}_unfrozen_{args.unfreeze_blocks}.pth"

    # --- TRAINING LOOP ---
    best_auc = 0.0
    patience = 7  # How many epochs to wait before giving up
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on Test Set after each epoch
        test_acc, test_auc, test_sens, test_spec = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} | Test AUC: {test_auc:.3f} | Acc: {test_acc*100:.1f}% | Sens: {test_sens*100:.1f}% | Spec: {test_spec*100:.1f}%")

        # --- EARLY STOPPING & SAVING ---
        if test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0  
            
            # Save the winning model weights
            torch.save(model.state_dict(), save_name)
            print(f"New best model saved. (AUC: {best_auc:.3f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
            
        # Pull the plug if it is overfitting
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered. The model stopped improving after {epoch+1} epochs.")
            break

    print(f"\nFinished. The best {args.model.upper()} model achieved an AUC of {best_auc:.3f}")
    print(f"Model saved as: {save_name}")

if __name__ == "__main__":
    main()