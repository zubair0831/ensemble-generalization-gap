import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score, average_precision_score
)
from sklearn.datasets import load_breast_cancer, load_wine

# Modern Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------
# 1. Model Builders
# -----------------------------------------------------------------------------

def get_models(n_classes=2, scale_pos_weight=1.0):
    """
    Returns a dictionary of models.
    scale_pos_weight: Used for XGB/LGB/CatBoost on imbalanced data.
    """
    random_state = 42
    
    # Base learners for Stacking
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ('svr', Pipeline([('sc', StandardScaler()), ('svm', SVC(probability=True))]))
    ]

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight='balanced' if scale_pos_weight > 1 else None))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight='balanced' if scale_pos_weight > 1 else None))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight='balanced' if scale_pos_weight > 1 else None),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight='balanced' if scale_pos_weight > 1 else None),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight='balanced' if scale_pos_weight > 1 else None),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=random_state, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, 
            scale_pos_weight=scale_pos_weight, objective="binary" if n_classes==2 else "multiclass",
            n_jobs=-1, random_state=random_state, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=4,
            scale_pos_weight=scale_pos_weight, verbose=False, random_state=random_state,
            allow_writing_files=False
        ),
        "Stacking": StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            n_jobs=-1
        )
    }
    return models

# -----------------------------------------------------------------------------
# 2. Cross-Validation Engine
# -----------------------------------------------------------------------------

def run_cv_experiment(name, X, y, n_splits=5, is_imbalanced_fraud=False):
    print(f"\n[{name}] Starting {n_splits}-Fold CV...")
    
    # Calculate scale_pos_weight for imbalance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    scale_pos_weight = n_neg / n_pos if is_imbalanced_fraud else 1.0
    
    if is_imbalanced_fraud:
        print(f"   -> Imbalanced Mode. Pos Ratio: 1:{int(scale_pos_weight)}")
    
    models = get_models(n_classes=len(np.unique(y)), scale_pos_weight=scale_pos_weight)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []

    for model_name, model in models.items():
        # Storage for fold metrics
        fold_train_acc = []
        fold_test_acc = []
        fold_f1 = []
        fold_auc = []
        fold_auprc = [] # Area Under Precision Recall Curve
        
        start_time = time()
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            
            # Proba (if available)
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        y_prob = y_prob[:, 1] # Binary case
                    else:
                        y_prob = y_prob # Multiclass case (AUC needs special handling, omitted for brevity if binary)
                except:
                    y_prob = None
            else:
                y_prob = None
            
            # Metrics
            fold_train_acc.append(accuracy_score(y_train, y_train_pred))
            fold_test_acc.append(accuracy_score(y_test, y_pred))
            
            # F1 (Macro for multiclass, Binary for binary)
            avg_method = 'binary' if len(np.unique(y)) == 2 else 'macro'
            fold_f1.append(f1_score(y_test, y_pred, average=avg_method))
            
            if y_prob is not None and len(np.unique(y)) == 2:
                fold_auc.append(roc_auc_score(y_test, y_prob))
                fold_auprc.append(average_precision_score(y_test, y_prob))
            else:
                fold_auc.append(0)
                fold_auprc.append(0)

        # Aggregate Results
        mean_test_acc = np.mean(fold_test_acc)
        std_test_acc = np.std(fold_test_acc)
        mean_train_acc = np.mean(fold_train_acc)
        gap = mean_train_acc - mean_test_acc
        
        results.append({
            "Model": model_name,
            "Test Acc Mean": mean_test_acc,
            "Test Acc Std": std_test_acc,
            "Train Acc Mean": mean_train_acc,
            "Gen Gap": gap,
            "F1 Mean": np.mean(fold_f1),
            "F1 Std": np.std(fold_f1),
            "AUC Mean": np.mean(fold_auc),
            "AUPRC Mean": np.mean(fold_auprc),
            "Time": time() - start_time
        })
        
    df = pd.DataFrame(results)
    print(f"[{name}] Done.")
    
    # Save raw data
    df.to_csv(f"results_{name}_cv.csv", index=False)
    
    # Generate Plots
    plot_cv_results(df, name, is_imbalanced_fraud)
    return df

# -----------------------------------------------------------------------------
# 3. Visualization
# -----------------------------------------------------------------------------

def plot_cv_results(df, dataset_name, is_fraud):
    sns.set_style("whitegrid")
    
    # Sort by performance
    sort_metric = "AUPRC Mean" if is_fraud else "Test Acc Mean"
    df = df.sort_values(sort_metric, ascending=False)
    
    # 1. Performance Plot with Error Bars
    plt.figure(figsize=(12, 6))
    
    if is_fraud:
        # For Fraud, we plot AUPRC
        plt.bar(df["Model"], df["AUPRC Mean"], yerr=0, capsize=5, color='salmon', alpha=0.8)
        plt.ylabel("Area Under Precision-Recall Curve (AUPRC)")
        plt.title(f"{dataset_name}: AUPRC Performance (5-Fold CV)")
    else:
        # For others, Test Accuracy with Std Dev error bars
        plt.bar(df["Model"], df["Test Acc Mean"], yerr=df["Test Acc Std"], capsize=5, color='skyblue', alpha=0.8)
        plt.ylabel("Test Accuracy")
        plt.title(f"{dataset_name}: Test Accuracy (5-Fold CV)")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_performance.png", dpi=300)
    plt.close()

    # 2. Generalization Gap Plot
    if not is_fraud:
        plt.figure(figsize=(12, 6))
        # Sort by Gap size for clarity
        df_gap = df.sort_values("Gen Gap", ascending=True) 
        plt.bar(df_gap["Model"], df_gap["Gen Gap"], color='thistle', edgecolor='black')
        plt.ylabel("Gap (Train Acc - Test Acc)")
        plt.title(f"{dataset_name}: Generalization Gap (Lower is better)")
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_gap.png", dpi=300)
        plt.close()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Breast Cancer
    data = load_breast_cancer()
    run_cv_experiment("BreastCancer", data.data, data.target)

    # 2. Heart Disease (Load your CSV)
    try:
        df_heart = pd.read_csv("heart.csv")
        X_h = df_heart.drop("target", axis=1).values
        y_h = df_heart["target"].values
        run_cv_experiment("HeartDisease", X_h, y_h)
    except FileNotFoundError:
        print("Skipping Heart Disease (csv not found)")

    # 3. Pima Diabetes (Load your CSV)
    try:
        df_pima = pd.read_csv("diabetes.csv")
        X_p = df_pima.drop("Outcome", axis=1).values
        y_p = df_pima["Outcome"].values
        run_cv_experiment("PimaDiabetes", X_p, y_p)
    except FileNotFoundError:
        print("Skipping Pima (csv not found)")

    # 4. Credit Card Fraud (Load your CSV)
    try:
        # Sampling for speed if dataset is huge, otherwise use full
        df_credit = pd.read_csv("creditcard.csv")
        # Optional: Subsample for testing the code quickly
        # df_credit = df_credit.sample(20000, random_state=42) 
        X_c = df_credit.drop("Class", axis=1).values
        y_c = df_credit["Class"].values
        run_cv_experiment("CreditCardFraud", X_c, y_c, is_imbalanced_fraud=True)
    except FileNotFoundError:
        print("Skipping Credit Fraud (csv not found)")