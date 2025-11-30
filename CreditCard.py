# creditcard_fraud_ensembles.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ===========================
# 1. Load dataset
# ===========================

# Make sure creditcard.csv is in the same folder as this script
df = pd.read_csv("creditcard.csv")

print("Columns:", df.columns.tolist())
print(df["Class"].value_counts())

X = df.drop("Class", axis=1).values
y = df["Class"].values  # 0 = non-fraud, 1 = fraud

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
pos_weight = n_neg / n_pos  # for XGBoost / LightGBM / CatBoost

print("\n=== Training set class distribution ===")
print("Non-fraud (0):", n_neg)
print("Fraud (1):    ", n_pos)
print("Imbalance ratio (neg:pos) ~", round(pos_weight, 1))


# ===========================
# 2. Define models
# ===========================

# Many models benefit from scaling in this dataset
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=5000,
        class_weight="balanced"    # handle imbalance
    ))
])

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5))
])



# Base tree (we still allow it to overfit)
tree = DecisionTreeClassifier(
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ),
    n_estimators=100,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

extra_trees = ExtraTreesClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=2,
        class_weight="balanced",
        random_state=42
    ),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

gboost = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# XGBoost, LightGBM, CatBoost with class imbalance handling
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    scale_pos_weight=pos_weight,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary",
    class_weight=None,
    scale_pos_weight=pos_weight,
    n_jobs=-1,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    loss_function="Logloss",
    scale_pos_weight=pos_weight,
    verbose=False,
    random_state=42
)

# Stacking (we reuse scaled LR + SVM; tree without scaling is fine)
stack_estimators = [
    ("lr", log_reg),
    ("tree", DecisionTreeClassifier(
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ))
]

stacking = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(
        max_iter=5000,
        class_weight="balanced"
    ),
    passthrough=True
)

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": tree,
    "KNN": knn,
    "Bagging (Trees)": bagging,
    "Random Forest": rf,
    "Extra Trees": extra_trees,
    "AdaBoost": ada,
    "Gradient Boosting": gboost,
    "XGBoost": xgb,
    "LightGBM": lgbm,
    "CatBoost": cat,
    "Stacking (LR+Tree+SVM)": stacking
}


# ===========================
# 3. Train + evaluate
# ===========================

results = []

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for ROC/PR AUC (if available)
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fall back: use decision_function if needed; here we just skip AUC
        y_test_proba = None

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = train_acc - test_acc

    f1 = f1_score(y_test, y_test_pred)
    recall_pos = recall_score(y_test, y_test_pred)  # recall for fraud class (1)

    if y_test_proba is not None:
        roc_auc = roc_auc_score(y_test, y_test_proba)
        pr_auc = average_precision_score(y_test, y_test_proba)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Generalization Gap": gap,
        "F1 (Fraud=1)": f1,
        "Recall (Fraud=1)": recall_pos,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=["Recall (Fraud=1)", "F1 (Fraud=1)"],
    ascending=False
).reset_index(drop=True)

pd.set_option("display.precision", 4)

print("\n=== Results on Credit Card Fraud Dataset ===")
print(results_df)
