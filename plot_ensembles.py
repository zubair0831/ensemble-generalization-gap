import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============== BASIC PER-DATASET PLOTS ===============

def plot_train_test_accuracy(results_df, dataset_name):
    """Single figure: Train vs Test Accuracy (bars) for one dataset."""
    models = results_df["Model"].values
    train_accs = results_df["Train Accuracy"].values
    test_accs = results_df["Test Accuracy"].values

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, train_accs, width, label="Train")
    ax.bar(x + width/2, test_accs, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.05)  # tweak if needed per dataset
    ax.set_title(f"{dataset_name}: Train vs Test Accuracy")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fname = f"{dataset_name.lower()}_acc.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_gap(results_df, dataset_name):
    """Single figure: Generalization Gap for one dataset."""
    models = results_df["Model"].values
    gaps = results_df["Generalization Gap"].values

    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, gaps)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Train - Test Accuracy")
    ax.set_title(f"{dataset_name}: Generalization Gap")

    plt.tight_layout()
    fname = f"{dataset_name.lower()}_gap.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_f1(results_df, dataset_name, f1_col="F1 Score (Test)"):
    """Single figure: F1 (or fraud F1) for one dataset."""
    models = results_df["Model"].values
    f1s = results_df[f1_col].values

    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, f1s)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(f1_col)
    ax.set_title(f"{dataset_name}: {f1_col} by Model")

    plt.tight_layout()
    fname = f"{dataset_name.lower()}_f1.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# =============== OPTIONAL: PIMA MULTI-SEED ERROR BARS ===============

def plot_pima_multiseed_errorbars(summary_csv, dataset_name="PimaDiabetes"):
    """
    Expects a CSV like your 'Multi-seed summary on PimaDiabetes',
    with columns:
      - 'Model'
      - 'Train Accuracy mean', 'Train Accuracy std'
      - 'Test Accuracy mean',  'Test Accuracy std'
      - 'Generalization Gap mean', 'Generalization Gap std'
      - 'F1 Score (Test) mean', 'F1 Score (Test) std'
    """
    df = pd.read_csv(summary_csv)
    models = df.index if "Model" not in df.columns else df["Model"].values

    # If "Model" is a column and not index:
    if "Model" in df.columns:
        df = df.set_index("Model")
        models = df.index.values

    x = np.arange(len(models))

    # (A) Test accuracy ± std
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        x,
        df["Test Accuracy mean"],
        yerr=df["Test Accuracy std"],
        fmt="o",
        capsize=4
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy (mean ± std)")
    ax.set_title(f"{dataset_name}: Test Accuracy Stability Across Seeds")
    plt.tight_layout()
    fname = f"{dataset_name.lower()}_multiseed_testacc.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    # (B) Generalization gap ± std
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        x,
        df["Generalization Gap mean"],
        yerr=df["Generalization Gap std"],
        fmt="o",
        capsize=4
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Gap (mean ± std)")
    ax.set_title(f"{dataset_name}: Generalization Gap Stability Across Seeds")
    plt.tight_layout()
    fname = f"{dataset_name.lower()}_multiseed_gap.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# =============== OPTIONAL: DATASET COMPLEXITY PLOT ===============

def plot_dataset_complexity(complexity_csv):
    """
    Expects your 'dataset complexity metrics' CSV with columns like:
      - Dataset
      - Linearity Score (SVMlin/SVMrbf)
      - Mean |Feature Corr|
      - Fisher Ratio
      - Intrinsic Dim (95% PCA)
      - Noise Estimate (LR CV std)
    We make one figure with separate subplots per metric,
    but each metric is readable on its own.
    """
    df = pd.read_csv(complexity_csv)

    datasets = df["Dataset"].values
    x = np.arange(len(datasets))

    metrics = [
        ("Linearity Score (SVMlin/SVMrbf)", "Linearity Score"),
        ("Mean |Feature Corr|", "Mean |Feature Corr|"),
        ("Fisher Ratio", "Fisher Ratio"),
        ("Intrinsic Dim (95% PCA)", "Intrinsic Dim (95% PCA)"),
        ("Noise Estimate (LR CV std)", "Noise Estimate (LR CV std)"),
    ]

    for col, ylabel in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, df[col].values)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by Dataset")
        plt.tight_layout()
        fname = f"complexity_{col.split('(')[0].strip().lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


# =============== MAIN DRIVER ===============

if __name__ == "__main__":
    # Map: dataset name -> (csv file, F1 column name)
    configs = [
        ("BreastCancer",    "results_breastcancer_cv.csv",          "F1 Score (Test)"),
        ("Wine",            "results_wine_cv.csv",                  "F1 Score (Test)"),
        ("HeartDisease",    "results_heartdisease_cv.csv",          "F1 Score (Test)"),
        ("PimaDiabetes",    "results_pimadiabetes_cv.csv",          "F1 Score (Test)"),
        ("CreditCardFraud", "results_creditcard_fraud_fast_cv.csv", "F1 (Fraud=1)"),
    ]

    for ds_name, csv_file, f1_col in configs:
        try:
            df = pd.read_csv(csv_file)
            print(f"\nLoaded {csv_file} for {ds_name}")

            # One figure = one clear message
            plot_train_test_accuracy(df, ds_name)
            plot_gap(df, ds_name)
            plot_f1(df, ds_name, f1_col=f1_col)

        except FileNotFoundError:
            print(f"WARNING: {csv_file} not found, skipping {ds_name}")

    # OPTIONAL: multi-seed Pima summary (if you saved it)
    # Example CSV name – change to your actual file:
    try:
        plot_pima_multiseed_errorbars("pimadiabetes_multiseed_summary.csv")
    except FileNotFoundError:
        print("No multi-seed summary CSV found, skipping multi-seed plots.")

    # OPTIONAL: dataset complexity plot (if you saved it)
    try:
        plot_dataset_complexity("dataset_complexity_metrics.csv")
    except FileNotFoundError:
        print("No dataset complexity CSV found, skipping complexity plots.")
