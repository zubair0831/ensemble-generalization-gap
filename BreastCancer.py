# ensemble_linear_datasets.py

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


def get_models(n_classes: int):
    """
    Build and return all models given number of classes.
    Automatically configures XGBoost/LightGBM/CatBoost for binary or multiclass.
    """
    is_multiclass = n_classes > 2

    # Pipelines for models that need scaling
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

    # Base decision tree (intentionally high capacity to show overfitting)
    tree = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    # Bagging and classic tree ensembles
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

    # Gradient boosting libs: configure according to class count
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

    # Stacking ensemble
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


def run_experiment(dataset_name: str, X: np.ndarray, y: np.ndarray):
    """
    Train all models on a given dataset and return a results DataFrame.
    Handles binary vs multiclass automatically.
    """
    print(f"\n=== Running on {dataset_name} ===")
    print("Features shape:", X.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2

    models = get_models(n_classes)
    results = []

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        if is_multiclass:
            f1 = f1_score(y_test, y_test_pred, average="weighted")
        else:
            f1 = f1_score(y_test, y_test_pred)

        gap = train_acc - test_acc

        results.append({
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Generalization Gap": gap,
            "F1 Score (Test)": f1
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)

    pd.set_option("display.precision", 4)
    print(f"\n=== Results on {dataset_name} ===")
    print(results_df)

    return results_df


if __name__ == "__main__":
    # 1) Breast Cancer – binary, near-linear
    bc = load_breast_cancer()
    bc_results = run_experiment("Breast Cancer", bc.data, bc.target)

    # 2) Wine – 3-class, relatively well-behaved
    wine = load_wine()
    wine_results = run_experiment("Wine", wine.data, wine.target)
