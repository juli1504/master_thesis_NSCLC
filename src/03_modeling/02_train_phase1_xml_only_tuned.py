"""
Phase 1b: Global Baselines (Hyperparameter Tuned)

This script uses GridSearchCV to automatically experiment with different
hyperparameters for MLP and XGBoost. It finds the optimal settings 
using K-Fold cross-validation on the SMOTE-balanced training data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def evaluate_model(name, model, X_test, y_test):
    """Calculates standard medical metrics for a given model."""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "Model": name,
        "Accuracy": f"{acc * 100:.1f}%",
        "AUC": f"{auc:.3f}",
        "Sensitivity": f"{sensitivity * 100:.1f}%",
        "Specificity": f"{specificity * 100:.1f}%"
    }

def main():
    print("Loading data for Hyperparameter Tuning...")
    
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    train_mask = df['dataset_split'] == 'Train'
    test_mask = df['dataset_split'] == 'Test'
    
    X_train_raw = X_encoded[train_mask]
    y_train = df.loc[train_mask, 'target']
    
    X_test_raw = X_encoded[test_mask]
    y_test = df.loc[test_mask, 'target']
    
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    X_train = scaler.fit_transform(X_train_imputed)
    X_test = scaler.transform(X_test_imputed)
    
    # --- 3. SMOTE UPSAMPLING ---
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # --- 4. EXPERIMENT GRIDS ---
    models = {
        "Tuned MLP (Neural Net)": (
            MLPClassifier(max_iter=1000, random_state=42), 
            {
                'hidden_layer_sizes': [(16,), (32, 16), (64, 32)],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.01] # Helps prevent overfitting
            }
        ),
        "Tuned XGBoost": (
            XGBClassifier(eval_metric='logloss', random_state=42), 
            {
                'max_depth': [2, 3, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100],
                'subsample': [0.7, 1.0] 
            }
        )
    }
    
    # --- 5. AUTOMATED TRAINING & TUNING ---
    results = []
    print("\nStarting Automated Grid Search (This will test dozens of combinations)...")
    for name, (model, param_grid) in models.items():
        print(f"Tuning {name}...")
        
        # cv=5 means 5-Fold Cross Validation. n_jobs=-1 uses all your laptop's CPU cores.
        gs = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        gs.fit(X_train_balanced, y_train_balanced)
        
        best_model = gs.best_estimator_
        
        # Evaluate the absolute best model on the unseen Test Set
        metrics = evaluate_model(name, best_model, X_test, y_test)
        results.append(metrics)
        print(f"Finished. Best settings found: {gs.best_params_}")
        
    # --- 6. DISPLAY RESULTS ---
    print("\n" + "="*75)
    print("PHASE 1 RESULTS: TUNED CLINICAL BASELINES")
    print("="*75)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    main()