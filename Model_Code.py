"""
kidney_disease_optimizer.py

This is a specialized script focused solely on achieving the highest possible accuracy
on the kidney disease dataset. It uses advanced imputation, extensive hyperparameter tuning,
and a stacking ensemble.
UPDATED: Hyperparameters for LightGBM have been expanded for a more flexible search.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ---------- Config ----------
DATA_PATH = "kidney_disease 3.csv"
TARGET_COL = "classification"
TARGET_MAP = {"ckd": 1, "notckd": 0}

BASE_OUTPUT_DIR = "results_kidney_optimized"
RANDOM_SEED = 42
CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 150
# ----------------------------

def main():
    """Main function to run the experiment."""
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"----- Processing Specialized Run for: {DATA_PATH} -----")

    # 1) Load and Preprocess Data
    try:
        df = pd.read_csv(DATA_PATH, na_values=['?', '\t?'])
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found: {DATA_PATH}.")
        print("Please ensure the script and the CSV file are in the same folder.")
        return

    if 'id' in df.columns: df = df.drop('id', axis=1)

    if TARGET_COL not in df.columns:
        print(f"FATAL ERROR: Target column '{TARGET_COL}' not found.")
        return

    df['target'] = df[TARGET_COL].map(TARGET_MAP)
    df.drop(columns=[TARGET_COL], inplace=True)
    df.dropna(subset=['target'], inplace=True)
    y = df['target'].astype(int)
    X_features = df.drop(columns=['target'])

    # One-hot encode categorical features
    X_features = pd.get_dummies(X_features, dummy_na=True, drop_first=True)
    feature_names = X_features.columns.tolist()

    # Impute and Scale
    print("Performing advanced imputation and scaling...")
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_features)
    X_std = scaler.fit_transform(X_imputed)
    X_std = pd.DataFrame(X_std, columns=feature_names)
    print("Preprocessing complete.")

    # 2) Define Classifiers & Hyperparameter Distributions
    models_to_tune = {
        "LogisticRegression": {
            "estimator": LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', max_iter=1000),
            "param_distributions": {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        },
        "XGBoost": {
            "estimator": xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', use_label_encoder=False),
            "param_distributions": {'n_estimators': [100, 200, 300, 400], 'learning_rate': [0.01, 0.05, 0.1, 0.15], 'max_depth': [3, 5, 7, 9], 'subsample': [0.6, 0.7, 0.8, 1.0], 'colsample_bytree': [0.6, 0.7, 0.8, 1.0]}
        },
        # *** UPDATED THIS SECTION ***
        "LightGBM": {
            "estimator": lgb.LGBMClassifier(random_state=RANDOM_SEED),
            "param_distributions": {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'num_leaves': [20, 31, 40, 50, 60],
                'subsample': [0.6, 0.7, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
                'min_child_samples': [5, 10, 20]
            }
        }
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results_df = []
    best_estimators = {}

    # 3) Tune each model
    for name, model_info in models_to_tune.items():
        print(f"Tuning: {name}...")
        random_search = RandomizedSearchCV(estimator=model_info["estimator"], param_distributions=model_info["param_distributions"], n_iter=N_ITER_RANDOM_SEARCH, cv=skf, scoring='accuracy', random_state=RANDOM_SEED, n_jobs=-1)
        random_search.fit(X_std, y)
        best_estimators[name] = random_search.best_estimator_
        results_df.append({"classifier": name, "accuracy_mean": random_search.best_score_, "best_params": random_search.best_params_})
        print(f"  Best Accuracy for {name}: {random_search.best_score_ * 100:.2f}%")

    # 4) Evaluate Stacking Ensemble
    print("\nBuilding and Evaluating Stacking Ensemble...")
    if all(name in best_estimators for name in ["LogisticRegression", "XGBoost", "LightGBM"]):
        estimators = [('lr', best_estimators["LogisticRegression"]), ('xgb', best_estimators["XGBoost"]), ('lgbm', best_estimators["LightGBM"])]
        meta_classifier = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear')
        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_classifier, cv=skf, n_jobs=-1)
        stacking_scores = cross_val_score(stacking_clf, X_std, y, cv=skf, scoring='accuracy', n_jobs=-1)
        stacking_mean_accuracy = stacking_scores.mean()
        results_df.append({"classifier": "StackingEnsemble (LR+XGB+LGBM)", "accuracy_mean": stacking_mean_accuracy, "best_params": "N/A"})
        print(f"  Stacking Ensemble Mean Accuracy: {stacking_mean_accuracy * 100:.2f}%")
        stacking_clf.fit(X_std, y)
        joblib.dump(stacking_clf, os.path.join(BASE_OUTPUT_DIR, "stacking_ensemble_best.joblib"))

    # 5) Save and print summary
    results_df = pd.DataFrame(results_df)
    print("\n--- Final Optimized Accuracy Summary for Kidney Disease ---")
    summary_df = results_df[['classifier', 'accuracy_mean']].copy()
    summary_df['accuracy_mean'] = (summary_df['accuracy_mean'] * 100).map('{:.2f}%'.format)
    summary_df.rename(columns={'accuracy_mean': 'Accuracy'}, inplace=True)
    print(summary_df.to_string(index=False))
    print(f"\nDone. Final results are in: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
