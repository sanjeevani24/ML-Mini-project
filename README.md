
Optimized Chronic Kidney Disease (CKD) Prediction using Stacking Ensemble
This repository contains the implementation of a specialized machine learning pipeline (kidney_disease_optimizer.py) focused on achieving the highest possible classification accuracy for the Chronic Kidney Disease (CKD) dataset.

The core approach involves advanced data preprocessing, aggressive hyperparameter tuning of high-performance models, and a final Stacking Ensemble to maximize predictive power.

Project Goal:
To build a predictive model that surpasses the accuracy benchmarks established by traditional and research-level machine learning approaches for CKD diagnosis, particularly by leveraging model diversity and optimization.

Methodology: Optimization Pipeline
The high accuracy of this approach is attributed to a combination of advanced techniques across the entire machine learning workflow:

1. Advanced Data Preprocessing
Missing Value Imputation: Uses K-Nearest Neighbors (KNN) Imputer (n_neighbors=5). This is a robust method that preserves relationships and feature variance, unlike simpler mean imputation.

Feature Scaling: Employs StandardScaler to normalize features, ensuring all features contribute equally to the distance metrics and stabilizing ensemble training.

Feature Encoding: Uses One-Hot Encoding to convert all nominal/categorical features into a binary numerical format suitable for modeling.

2. Base Model Tuning and Selection
Three high-performing, diverse models are selected and rigorously optimized to serve as the base learners:

XGBoost: Optimized via RandomizedSearchCV (150 iterations). Example parameters tuned: learning_rate, max_depth, n_estimators.

LightGBM: Optimized via RandomizedSearchCV (150 iterations). Example parameters tuned: num_leaves, min_child_samples, learning_rate.

Logistic Regression: Optimized via RandomizedSearchCV (150 iterations). Example parameters tuned: C (regularization strength), penalty (l1/l2).

3. Final Model: Stacking Ensemble
The Stacking Ensemble is the final predictive layer:

Base Estimators: The three optimized models (Logistic Regression, XGBoost, and LightGBM).

Meta-Classifier: Logistic Regression.

Validation: All optimization and final evaluation steps use 5-fold Stratified Cross-Validation to ensure robust and unbiased accuracy reporting.

Key Improvements over Research Paper Baseline:
Compared to standard research approaches (e.g., Cost-Sensitive AdaBoost with Mean Imputation and Information Gain Feature Selection), this pipeline achieves higher accuracy through:

Superior Imputation: KNN Imputation provides higher data quality than Mean Imputation.

Modern Boosted Models: Utilizing XGBoost and LightGBM as base learners, which are generally more powerful and accurate than the Decision Tree base used in classical AdaBoost.

Rigorously Optimized Parameters: The extensive tuning process ensures each model is operating at its maximum performance threshold.

Powerful Ensemble: Stacking optimally combines the strengths of diverse models, leading to a performance ceiling often higher than any single individual model. 
