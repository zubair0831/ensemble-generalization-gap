import matplotlib.pyplot as plt
from BreastCancer import results_df

models_list = results_df["Model"]
train_accs = results_df["Train Accuracy"]
test_accs = results_df["Test Accuracy"]

plt.figure(figsize=(12,6))
plt.bar(results_df["Model"], results_df["F1 Score (Test)"])
plt.xticks(rotation=75)
plt.title("F1 Score Comparison Across Models")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()
