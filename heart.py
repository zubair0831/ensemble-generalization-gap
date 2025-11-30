# tabular_ensembles_all.py

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine
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

from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ---------------------------
# Build model zoo
# ---------------------------

def get_models(n_classes: int):
    """Return dict of all models. Handles binary vs multiclass."""
    is_multiclass = n_classes > 2

    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    svm_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

    tree = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=200,
        random_state=42
    )

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )

    gboost = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    # XGBoost / LightGBM / CatBoost: config for binary or multiclass
    if is_multiclass:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42
        )

        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            num_class=n_classes,
            n_jobs=-1,
            random_state=42
        )

        cat = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="MultiClass",
            verbose=False,
            random_state=42
        )
    else:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )

        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            n_jobs=-1,
            random_state=42
        )

        cat = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="Logloss",
            verbose=False,
            random_state=42
        )

    base_estimators = [
        ("lr", log_reg),
        ("tree", DecisionTreeClassifier(random_state=42)),
        ("svm", svm_rbf)
    ]
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        passthrough=True
    )

    models = {
        "Logistic Regression": log_reg,
        "Decision Tree": tree,
        "KNN": knn,
        "SVM (RBF)": svm_rbf,
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

    return models


# ---------------------------
# Run one dataset
# ---------------------------

def run_experiment(name: str, X, y):
    print(f"\n=== Running on {name} ===")
    print("Features shape:", X.shape)
    uniq, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(uniq, counts)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2

    models = get_models(n_classes)
    rows = []

    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(X_train, y_train)

        y_tr = model.predict(X_train)
        y_te = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_tr)
        test_acc = accuracy_score(y_test, y_te)

        if is_multiclass:
            f1 = f1_score(y_test, y_te, average="weighted")
        else:
            f1 = f1_score(y_test, y_te)

        gap = train_acc - test_acc

        rows.append({
            "Model": model_name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Generalization Gap": gap,
            "F1 Score (Test)": f1
        })

    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)

    pd.set_option("display.precision", 4)
    print(f"\n=== Results on {name} ===")
    print(df_res)

    return df_res


# ---------------------------
# Main: run all tabular datasets
# ---------------------------

if __name__ == "__main__":
    # 1) Breast Cancer (binary, near-linear)
    bc = load_breast_cancer()
    bc_results = run_experiment("Breast Cancer", bc.data, bc.target)

    # 2) Wine (3-class, moderately nonlinear)
    wine = load_wine()
    wine_results = run_experiment("Wine", wine.data, wine.target)

    # 3) Heart Disease (binary, medical, moderate complexity)
    heart_df = pd.read_csv("heart.csv")      # Kaggle UCI Heart
    X_heart = heart_df.drop("target", axis=1).values
    y_heart = heart_df["target"].values
    heart_results = run_experiment("Heart Disease", X_heart, y_heart)

    # 4) Diabetes (binary, medical, noisy)
    diab_df = pd.read_csv("diabetes.csv")    # Kaggle Pima Indians Diabetes
    X_diab = diab_df.drop("Outcome", axis=1).values
    y_diab = diab_df["Outcome"].values
    diab_results = run_experiment("Pima Diabetes", X_diab, y_diab)
bc_results.to_csv("breast_cancer_results.csv", index=False)
wine_results.to_csv("wine_results.csv", index=False)
heart_results.to_csv("heart_results.csv", index=False)
diab_results.to_csv("pima_diabetes_results.csv", index=False)
fraud_results.to_csv("credit_fraud_results.csv", index=False)
