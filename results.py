import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 讀取 CSV
df = pd.read_csv("crops/results.csv")


y_true = df["gt_class"].astype(str)
y_pred = df["pred_class"].astype(str)


labels = sorted(set(y_true) | set(y_pred))


cm = confusion_matrix(y_true, y_pred, labels=labels)


plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca(), values_format="d")
plt.title("Confusion Matrix (Counts)")
plt.savefig("confusion_matrix_counts.png", dpi=300, bbox_inches="tight")
plt.close()


cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp_norm.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca(), values_format=".2f")
plt.title("Confusion Matrix (Normalized %)")
plt.savefig("confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
plt.close()

