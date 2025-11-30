# ensemble_experiments.py
# Cross-validated comparison of ensemble models with light hyperparameter tuning.
# Enhanced for MacBook Air M2:
#   - Multiple seeds on small datasets (mean ± std)
#   - Learning curves for 4 key models
#   - Dataset complexity metrics
#
# Requires: scikit-learn, xgboost, lightgbm, catboost, matplotlib, pandas, numpy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
    learning_curve,       # NEW
)
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

from sklearn.datasets import load_breast_cancer, load_wine

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

### NEW: seeds for repeated CV on small datasets
SEEDS = [42, 123, 456, 789, 1011]

# ============================================================
# Model builders
# ============================================================

def build_models_balanced(n_classes):
    """Return dict of models for balanced/binary datasets."""
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    svm_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

    tree = DecisionTreeClassifier(random_state=42)

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
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

    if n_classes == 2:
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
            random_state=42,
            verbose=-1   # reduce LightGBM logs
        )
        cat = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="Logloss",
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
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
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
            random_state=42,
            verbose=-1
        )
        cat = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="MultiClass",
            verbose=False,
            random_state=42
        )

    estimators = [
        ("lr", log_reg),
        ("tree", DecisionTreeClassifier(random_state=42)),
        ("svm", svm_rbf)
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=5000),
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


def build_models_imbalanced(pos_weight):
    """Models for Credit Card Fraud with imbalance handling."""
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        ))
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

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
        n_estimators=200,
        random_state=42,
        n_jobs=-1
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

    xgb = XGBClassifier(
        n_estimators=300,
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
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        loss_function="Logloss",
        scale_pos_weight=pos_weight,
        verbose=False,
        random_state=42
    )

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
    return models


# ============================================================
# Hyperparameter grids
# ============================================================

def get_param_grids_balanced():
    return {
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0]
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 11]
        },
        "SVM (RBF)": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto"]
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10],
            "max_features": ["sqrt", "log2"]
        },
        "Extra Trees": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10],
            "max_features": ["sqrt", "log2"]
        },
        "Gradient Boosting": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3]
        },
        "XGBoost": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4],
        },
        "LightGBM": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [31, 63],
        },
        "CatBoost": {
            "iterations": [200, 300],
            "learning_rate": [0.03, 0.05],
            "depth": [4, 6]
        },
        # Bagging, AdaBoost, Stacking left without tuning
    }


def get_param_grids_imbalanced():
    return {
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0]
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 11]
        },
        "Decision Tree": {
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10],
            "max_features": ["sqrt", "log2"]
        },
        "Extra Trees": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10],
            "max_features": ["sqrt", "log2"]
        },
        "Gradient Boosting": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3]
        },
        "XGBoost": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4],
        },
        "LightGBM": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [31, 63],
        },
        "CatBoost": {
            "iterations": [200, 300],
            "learning_rate": [0.03, 0.05],
            "depth": [4, 6]
        },
        # Bagging, AdaBoost, Stacking left without tuning
    }


# ============================================================
# Helper: safe StratifiedKFold
# ============================================================

def make_safe_cv(y, max_splits=5, random_state=42):  # NEW: random_state param
    """Choose n_splits <= min_class_count to avoid 'cannot split' errors."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    n_splits = min(max_splits, int(min_count))
    if n_splits < 2:
        raise ValueError(
            f"Not enough samples per class for CV: min_class_count={min_count}"
        )
    print(f"Using StratifiedKFold with n_splits={n_splits}, random_state={random_state}")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ============================================================
# Helper: fit + CV + optional tuning (with fallback)
# ============================================================

def evaluate_with_cv(
    model_name,
    model,
    X,
    y,
    cv,
    is_binary=True,
    param_grid=None,
    is_imbalanced=False
):
    """
    Run CV (and hyperparameter tuning if param_grid is not None).
    If GridSearch fails (e.g., weird split issues), falls back to plain CV.
    """
    print(f"\n--- {model_name} ---")

    # Scorers
    if is_imbalanced:
        scoring = {
            "acc": "accuracy",
            "f1": "f1",
            "recall": "recall",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision"
        }
        refit_metric = "f1"
    else:
        if is_binary:
            scoring = {"acc": "accuracy", "f1": "f1"}
        else:
            scoring = {"acc": "accuracy", "f1": "f1_macro"}
        refit_metric = "f1"

    # Try GridSearchCV if param_grid is provided
    if param_grid is not None and len(param_grid) > 0:
        print("  Using GridSearchCV for tuning...")
        try:
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                refit=refit_metric,
                n_jobs=-1,
                return_train_score=True,
                error_score="raise"
            )
            grid.fit(X, y)
            best_idx = grid.best_index_
            cvres = grid.cv_results_

            mean_train_acc = cvres["mean_train_acc"][best_idx]
            mean_test_acc = cvres["mean_test_acc"][best_idx]
            mean_test_f1 = cvres[f"mean_test_{refit_metric}"][best_idx]
            gap = mean_train_acc - mean_test_acc

            metrics = {
                "Model": model_name,
                "Train Accuracy": mean_train_acc,
                "Test Accuracy": mean_test_acc,
                "Generalization Gap": gap,
            }

            if is_imbalanced:
                mean_test_recall = cvres["mean_test_recall"][best_idx]
                mean_test_roc = cvres["mean_test_roc_auc"][best_idx]
                mean_test_pr = cvres["mean_test_pr_auc"][best_idx]
                metrics.update({
                    "F1 (Fraud=1)": mean_test_f1,
                    "Recall (Fraud=1)": mean_test_recall,
                    "ROC-AUC": mean_test_roc,
                    "PR-AUC": mean_test_pr
                })
            else:
                metrics["F1 Score (Test)"] = mean_test_f1

            print("  Best params:", grid.best_params_)
            print("  Train Acc (CV): {:.4f}, Test Acc (CV): {:.4f}, Gap: {:.4f}".format(
                mean_train_acc, mean_test_acc, gap
            ))
            return metrics

        except Exception as e:
            print("  GridSearch failed for", model_name, "->", e)
            print("  Falling back to plain cross_validate with default params...")

    # Fallback or no tuning: plain CV
    print("  Running cross_validate without tuning...")
    cv_results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )

    mean_train_acc = cv_results["train_acc"].mean()
    mean_test_acc = cv_results["test_acc"].mean()
    gap = mean_train_acc - mean_test_acc

    metrics = {
        "Model": model_name,
        "Train Accuracy": mean_train_acc,
        "Test Accuracy": mean_test_acc,
        "Generalization Gap": gap,
    }

    if is_imbalanced:
        mean_test_f1 = cv_results["test_f1"].mean()
        mean_test_recall = cv_results["test_recall"].mean()
        mean_test_roc = cv_results["test_roc_auc"].mean()
        mean_test_pr = cv_results["test_pr_auc"].mean()
        metrics.update({
            "F1 (Fraud=1)": mean_test_f1,
            "Recall (Fraud=1)": mean_test_recall,
            "ROC-AUC": mean_test_roc,
            "PR-AUC": mean_test_pr
        })
    else:
        mean_test_f1 = cv_results["test_f1"].mean()
        metrics["F1 Score (Test)"] = mean_test_f1

    print("  Train Acc (CV): {:.4f}, Test Acc (CV): {:.4f}, Gap: {:.4f}".format(
        mean_train_acc, mean_test_acc, gap
    ))
    return metrics


# ============================================================
# NEW: Dataset complexity metrics
# ============================================================

def compute_dataset_complexity(X, y, dataset_name):
    """
    Quantify why datasets behave differently:
    - Linearity score: LR vs RBF SVM
    - Mean abs feature correlation
    - Fisher ratio (class separability, binary only)
    - Intrinsic dimensionality (PCA@95%)
    - Noise estimate: std of LR CV scores
    """
    print(f"\n[Complexity] Computing metrics for {dataset_name}...")

    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA

    # Standardize for linear models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_clf = LogisticRegression(max_iter=5000)
    svm_lin = SVC(kernel='linear')
    svm_rbf = SVC(kernel='rbf')

    lr_score = cross_val_score(lr_clf, X_scaled, y, cv=5, scoring='accuracy').mean()
    svm_lin_score = cross_val_score(svm_lin, X_scaled, y, cv=5, scoring='accuracy').mean()
    svm_rbf_score = cross_val_score(svm_rbf, X_scaled, y, cv=5, scoring='accuracy').mean()

    linearity_score = svm_lin_score / (svm_rbf_score + 1e-10)

    # Feature correlation
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
    if X_arr.shape[1] > 1:
        corr_matrix = np.corrcoef(X_arr.T)
        upper = np.triu_indices_from(corr_matrix, k=1)
        mean_abs_corr = np.abs(corr_matrix[upper]).mean()
    else:
        mean_abs_corr = 0.0

    # Fisher ratio (binary only)
    classes = np.unique(y)
    if len(classes) == 2:
        X0 = X_arr[y == classes[0]]
        X1 = X_arr[y == classes[1]]
        between_var = ((X0.mean(axis=0) - X1.mean(axis=0)) ** 2).sum()
        within_var = X0.var(axis=0).sum() + X1.var(axis=0).sum()
        fisher_ratio = between_var / (within_var + 1e-10)
    else:
        fisher_ratio = np.nan

    # Intrinsic dimensionality via PCA(95% variance)
    pca = PCA(n_components=0.95)
    pca.fit(X_arr)
    intrinsic_dim = pca.n_components_

    # Noise estimate: std of LR CV accuracy
    lr_scores = cross_val_score(lr_clf, X_scaled, y, cv=5, scoring='accuracy')
    noise_estimate = lr_scores.std()

    return {
        "Dataset": dataset_name,
        "N Samples": X_arr.shape[0],
        "N Features": X_arr.shape[1],
        "Linearity Score (SVMlin/SVMrbf)": linearity_score,
        "Mean |Feature Corr|": mean_abs_corr,
        "Fisher Ratio": fisher_ratio,
        "Intrinsic Dim (95% PCA)": intrinsic_dim,
        "Noise Estimate (LR CV std)": noise_estimate
    }


# ============================================================
# NEW: Learning curves for 4 representative models
# ============================================================

def plot_learning_curves(dataset_name, X, y):
    """
    Plot learning curves (train/test + gap) for:
      - Logistic Regression
      - Decision Tree
      - Random Forest
      - XGBoost
    Uses default-ish configs for speed.
    """
    print(f"\n[Learning Curves] {dataset_name}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
    }

    train_sizes = np.linspace(0.1, 1.0, 8)
    n_models = len(models)
    cols = 2
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        print(f"  LC for {name}...")

        train_sizes_abs, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=train_sizes,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)
        gap = train_mean - test_mean

        ax.plot(train_sizes_abs, train_mean, "o-", label="Train Acc")
        ax.plot(train_sizes_abs, test_mean, "o-", label="Test Acc")
        ax.fill_between(train_sizes_abs, test_mean, train_mean, alpha=0.15, label="Gap")
        ax.set_title(name)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Learning Curves - {dataset_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"{dataset_name.lower().replace(' ', '_')}_learning_curves.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Experiments: balanced datasets (multi-seed wrapper)
# ============================================================

def run_balanced_experiment_single_seed(name, X, y, seed):
    """Run your original balanced experiment for a single seed."""
    print("\n===================================================")
    print(f"=== Running CV + tuning on {name}, seed={seed} ===")
    print("Features shape:", X.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    n_classes = len(unique)
    models = build_models_balanced(n_classes)
    param_grids = get_param_grids_balanced()

    cv = make_safe_cv(y, max_splits=5, random_state=seed)

    results = []
    for model_name, model in models.items():
        pg = param_grids.get(model_name, None)
        metrics = evaluate_with_cv(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            cv=cv,
            is_binary=(n_classes == 2),
            param_grid=pg,
            is_imbalanced=False
        )
        metrics["Seed"] = seed  # NEW: track seed
        results.append(metrics)

    df = pd.DataFrame(results)
    return df


def run_balanced_experiment_multi_seed(name, X, y, seeds=SEEDS):
    """
    Run balanced experiment across multiple seeds, aggregate mean ± std.
    Writes:
      - per-seed raw CSV
      - aggregated summary CSV
      - gap/accuracy/F1 plots using mean values
    """
    all_dfs = []
    for seed in seeds:
        df_seed = run_balanced_experiment_single_seed(name, X, y, seed)
        df_seed.to_csv(
            f"results_{name.lower().replace(' ', '_')}_cv_seed{seed}.csv",
            index=False
        )
        all_dfs.append(df_seed)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Aggregate: mean ± std across seeds
    metric_cols = ["Train Accuracy", "Test Accuracy", "Generalization Gap", "F1 Score (Test)"]
    agg = {}
    for col in metric_cols:
        if col in df_all.columns:
            agg[col] = ["mean", "std"]

    summary = df_all.groupby("Model").agg(agg)
    summary.columns = [" ".join(col).strip() for col in summary.columns.values]
    summary = summary.sort_values(by="Test Accuracy mean", ascending=False)
    summary.to_csv(
        f"results_{name.lower().replace(' ', '_')}_cv_multi_seed_summary.csv"
        , index=True
    )

    print(f"\n=== Multi-seed summary on {name} ===")
    print(summary)

    # For plots, use mean values only
    df_mean = summary.reset_index()[["Model",
                                     "Train Accuracy mean",
                                     "Test Accuracy mean",
                                     "Generalization Gap mean",
                                     "F1 Score (Test) mean"]]
    df_mean = df_mean.rename(columns={
        "Train Accuracy mean": "Train Accuracy",
        "Test Accuracy mean": "Test Accuracy",
        "Generalization Gap mean": "Generalization Gap",
        "F1 Score (Test) mean": "F1 Score (Test)"
    })

    make_gap_plot(df_mean, f"{name} (multi-seed)")
    return df_all, summary


# ============================================================
# Experiment: Credit Card Fraud (FAST version, single seed)
# ============================================================

def run_creditcard_experiment(csv_path="creditcard.csv", frac=0.1):
    """
    Faster CV experiment for Credit Card Fraud:
    - Subsamples the dataset (default 10%)
    - Uses 3-fold StratifiedKFold
    - No GridSearchCV (only default hyperparameters)
    - Drops KNN and Stacking for speed
    """
    df = pd.read_csv(csv_path)
    print("\n===================================================")
    print("=== Running FAST CV on Credit Card Fraud ===")
    print("Original shape:", df.shape)

    # ---- 1) Subsample for speed ----
    df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
    print(f"Subsampled shape (frac={frac}):", df.shape)

    print(df["Class"].value_counts())

    # Use DataFrame/Series (not .values) to avoid LightGBM feature-name warnings
    X = df.drop("Class", axis=1)
    y = df["Class"]

    classes, counts = np.unique(y, return_counts=True)
    n_pos = counts[classes.tolist().index(1)]
    n_neg = counts[classes.tolist().index(0)]
    pos_weight = n_neg / n_pos

    print("\nOverall class distribution (after subsample)")
    print("Non-fraud (0):", n_neg)
    print("Fraud (1):    ", n_pos)
    print("Imbalance ratio (neg:pos) ~", round(pos_weight, 1))

    models = build_models_imbalanced(pos_weight)

    # Drop very slow models for this large dataset
    for slow_name in ["KNN", "Stacking (LR+Tree+SVM)"]:
        if slow_name in models:
            print(f"Removing slow model for CreditCard: {slow_name}")
            models.pop(slow_name)

    # We skip GridSearch for this huge dataset -> only cross_validate
    cv = make_safe_cv(y, max_splits=3, random_state=42)

    results = []
    for model_name, model in models.items():
        metrics = evaluate_with_cv(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            cv=cv,
            is_binary=True,
            param_grid=None,      # no GridSearchCV for fraud
            is_imbalanced=True
        )
        results.append(metrics)

    df_res = pd.DataFrame(results)
    # Sort primarily by Recall on Fraud=1 (focus of fraud detection)
    df_res = df_res.sort_values(
        by=["Recall (Fraud=1)"],
        ascending=False
    ).reset_index(drop=True)

    pd.set_option("display.precision", 4)
    print("\n=== FAST CV Results on Credit Card Fraud Dataset ===")
    print(df_res)

    df_res.to_csv("results_creditcard_fraud_fast_cv.csv", index=False)
    make_gap_plot(df_res, "Credit Card Fraud (FAST)", f1_col="F1 (Fraud=1)")

    return df_res, X, y


# ============================================================
# Plot helper
# ============================================================

def make_gap_plot(results_df, dataset_name, f1_col="F1 Score (Test)"):
    models = results_df["Model"].values
    train_accs = results_df["Train Accuracy"].values
    test_accs = results_df["Test Accuracy"].values
    gaps = results_df["Generalization Gap"].values
    f1s = results_df[f1_col].values

    x = np.arange(len(models))

    plt.figure(figsize=(12, 6))
    plt.plot(x, train_accs, marker="o", label="Train Accuracy (CV)")
    plt.plot(x, test_accs, marker="o", label="Test Accuracy (CV)")
    plt.xticks(x, models, rotation=75, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"Train vs Test Accuracy (CV) - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_acc_cv.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(x, gaps)
    plt.xticks(x, models, rotation=75, ha="right")
    plt.ylabel("Train - Test Accuracy (CV)")
    plt.title(f"Generalization Gap (CV) - {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_gap_cv.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(x, f1s)
    plt.xticks(x, models, rotation=75, ha="right")
    plt.ylabel(f1_col)
    plt.title(f"{f1_col} by Model (CV) - {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_f1_cv.png", dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    complexity_rows = []   # NEW: collect dataset complexity info

    # 1) Breast Cancer
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    bc_all, bc_summary = run_balanced_experiment_multi_seed("BreastCancer", X_bc, y_bc)
    complexity_rows.append(compute_dataset_complexity(X_bc, y_bc, "BreastCancer"))
    plot_learning_curves("BreastCancer", X_bc, y_bc)

    # 2) Wine (optional in paper, but can still run)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    wine_all, wine_summary = run_balanced_experiment_multi_seed("Wine", X_wine, y_wine)
    complexity_rows.append(compute_dataset_complexity(X_wine, y_wine, "Wine"))
    plot_learning_curves("Wine", X_wine, y_wine)

    # 3) Heart Disease
    heart_df = pd.read_csv("heart.csv")
    X_heart = heart_df.drop("target", axis=1).values
    y_heart = heart_df["target"].values
    heart_all, heart_summary = run_balanced_experiment_multi_seed("HeartDisease", X_heart, y_heart)
    complexity_rows.append(compute_dataset_complexity(X_heart, y_heart, "HeartDisease"))
    plot_learning_curves("HeartDisease", X_heart, y_heart)

    # 4) Pima Diabetes
    diab_df = pd.read_csv("diabetes.csv")
    X_diab = diab_df.drop("Outcome", axis=1).values
    y_diab = diab_df["Outcome"].values
    diab_all, diab_summary = run_balanced_experiment_multi_seed("PimaDiabetes", X_diab, y_diab)
    complexity_rows.append(compute_dataset_complexity(X_diab, y_diab, "PimaDiabetes"))
    plot_learning_curves("PimaDiabetes", X_diab, y_diab)

    # 5) Credit Card Fraud (FAST, subsample 10%, single seed)
    credit_results, X_credit, y_credit = run_creditcard_experiment("creditcard.csv", frac=0.1)
    complexity_rows.append(compute_dataset_complexity(X_credit.values, y_credit.values, "CreditCard(10%)"))

    # Save dataset complexity table
    df_complexity = pd.DataFrame(complexity_rows)
    df_complexity.to_csv("dataset_complexity_metrics.csv", index=False)
    print("\n=== Dataset complexity metrics ===")
    print(df_complexity)
