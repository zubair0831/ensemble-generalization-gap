Ensemble Generalization Gap – Bias–Variance Behaviour on Tabular Data

This repository contains the experimental code and result summaries used in the study:

“How Ensemble Learning Balances Accuracy and Overfitting:
A Bias–Variance Perspective on Tabular Data.”

The project investigates how different machine-learning models — from simple linear classifiers to modern tree-based ensembles — generalize across datasets that vary in linearity, noise, and class imbalance. The focus is on understanding the generalization gap (train–test discrepancy) as a practical indicator of overfitting.

Contents
ensemble-generalization-gap/
│
│  ensemble_experiments.py        ← main experiment script
│  dataset_complexity_metrics.csv ← computed dataset-complexity indicators
│  results_*multi_seed_summary.csv← aggregated mean ± std results (multi-seed CV)
│  results_creditcard_fraud_fast_cv.csv ← fraud experiment metrics
│
│  complexity_linearity_score.png
│  complexity_noise_estimate.png
│  complexity_mean_|feature_corr|.png
│  complexity_intrinsic_dim.png
│  complexity_fisher_ratio.png
│
└─ .gitignore


Raw datasets (heart.csv, diabetes.csv, etc.) are intentionally not included.
They should be downloaded from UCI/Kaggle if needed.

Running the Experiments

Python 3.10+ recommended.

Install dependencies:

pip install numpy pandas scikit-learn matplotlib xgboost lightgbm catboost


Then run:

python3 ensemble_experiments.py


This will:

run stratified cross-validation across multiple models,

compute dataset complexity metrics,

create summary CSV results,

generate learning-curve diagrams and diagnostic plots.

Methodology Summary

Models compared:

Logistic Regression

K-Nearest Neighbours

Support Vector Machine (RBF)

Decision Tree

Bagging

Random Forest

Extra Trees

AdaBoost

Gradient Boosting

XGBoost / LightGBM / CatBoost

Stacking

Key evaluation metrics:

Train & Test Accuracy

Accuracy-based Generalization Gap

F1 Score (macro for balanced datasets)

Fraud-class F1 / Recall / PR-AUC (for credit-card fraud)

Interpretation

We observe three regimes:

Clean / near-linear data → linear models generalize best

Structured nonlinear data → tree ensembles outperform with small gaps

Noisy / imbalanced data → large gaps appear; ensembles may overfit or miss the minority class

Generalization-gap visualizations and complexity metrics help determine when extra model capacity is helpful vs. harmful.

Citation

If you use this code or experimental framework, please cite:

Z. A. Mohammad,
“How Ensemble Learning Balances Accuracy and Overfitting:
A Bias–Variance Perspective on Tabular Data”, 2025.

Author

Zubair Ahmed Mohammad
CS Department, VIT-AP University
Email: zubairahmed20050831@gmail.com
