"""
Phase 1: Global Baselines (Clinical Metadata Only)

This script trains standard Machine Learning models (Logistic Regression, 
XGBoost, and a simple MLP) on the true clinical patient data (Age, Gender, Smoking).
It evaluates the baseline performance strictly on the untouched Test set 
to ensure a direct "apples-to-apples" comparison with the Tuned models.
Now updated to include F1-Score for imbalanced class performance.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def evaluate_model(name, model, X_eval, y_eval):
    """Calculates standard medical metrics plus F1-Score for a given model."""
    y_probs = model.predict_proba(X_eval)[:, 1]
    y_pred = model.predict(X_eval)
    
    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred) # Harmonic mean of Precision and Recall
    
    try:
        auc = roc_auc_score(y_eval, y_probs)
    except ValueError:
        auc = 0.500 
        
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "Model": name,
        "Accuracy": f"{acc * 100:.1f}%",
        "F1-Score": f"{f1:.3f}",
        "AUC": f"{auc:.3f}",
        "Sensitivity": f"{sensitivity * 100:.1f}%",
        "Specificity": f"{specificity * 100:.1f}%"
    }

def main():
    print("Loading manifest and clinical data...")
    
    # 1. Load both datasets
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    
    # 2. Filter out Excluded patients
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    
    # 3. Merge on Patient ID
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    print(f"Successfully merged data for {len(df)} valid patients!")

    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    print(f"Target Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")
    
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    # --- STRICT HOLDOUT SPLIT ---
    train_mask = df['dataset_split'] == 'Train'
    test_mask = df['dataset_split'] == 'Test' 
    
    X_train_raw = X_encoded[train_mask]
    y_train = df.loc[train_mask, 'target']
    
    X_test_raw = X_encoded[test_mask]
    y_test = df.loc[test_mask, 'target']
    
    imputer = SimpleImputer(strategy='median') 
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_test = scaler.transform(X_test_imputed)
    
    # --- 3. MODEL INITIALIZATION ---
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
        "Simple MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # --- 4. TRAINING & EVALUATION ---
    results = []
    print("Applying SMOTE upsampling to the Training Set...")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {np.bincount(y_train)}")
    print(f"Balanced training shape: {np.bincount(y_train_balanced)}")
    print(f"Test Set shape: {np.bincount(y_test)}")
    
    print("\nTraining models on balanced data...")
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        print(f"{name} finished.")
        
    # --- 5. DISPLAY RESULTS ---
    print("\n" + "="*85)
    print("PHASE 1 RESULTS: PURE CLINICAL DATA (Age, Gender, Smoking)")
    print("(Evaluated strictly on the untouched Test Set with F1-Score)")
    print("="*85)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()