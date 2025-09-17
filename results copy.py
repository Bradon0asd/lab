import pandas as pd
from sklearn.metrics import classification_report

# 讀取 CSV
df = pd.read_csv("crops/results.csv")

# ground truth & prediction
y_true = df["gt_class"].astype(str)
y_pred = df["pred_class"].astype(str)

# 取得報告 (dict 格式)
report_dict = classification_report(y_true, y_pred, output_dict=True)

# 取出整體數值
overall_metrics = {
    "accuracy": report_dict["accuracy"],
    "macro_precision": report_dict["macro avg"]["precision"],
    "macro_recall": report_dict["macro avg"]["recall"],
    "macro_f1": report_dict["macro avg"]["f1-score"],
    "weighted_precision": report_dict["weighted avg"]["precision"],
    "weighted_recall": report_dict["weighted avg"]["recall"],
    "weighted_f1": report_dict["weighted avg"]["f1-score"],
}

print("=== Overall Metrics ===")
for k, v in overall_metrics.items():
    print(f"{k}: {v:.4f}")
