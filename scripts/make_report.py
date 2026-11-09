"""Membuat tabel perbandingan Part 2 vs Part 3 dan grafik pendukung dari hasil training."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # path relatif terhadap folder proyek
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

m = json.load(open("reports/evaluation/metrics_pytorch.json"))
base = json.load(open("reports/evaluation/baseline_part2.json"))

def diag(train_auc, test_auc):
    gap = train_auc - test_auc
    return "Overfit" if gap > 0.05 else ("Underfit" if train_auc < 0.65 else "Good fit")

rows = []
for name, b in base.items():
    rows.append({"Model": f"{name} (Part 2)", "Accuracy": b["accuracy"], "Precision": b["precision"], "Recall": b["recall"],
                 "F1": b["f1"], "ROC-AUC": b["roc_auc"], "Train AUC": b["train_auc"],
                 "Gap AUC": b["train_auc"] - b["roc_auc"], "Diagnosis": diag(b["train_auc"], b["roc_auc"]),
                 "TN": b["confusion_matrix"][0][0], "FP": b["confusion_matrix"][0][1],
                 "FN": b["confusion_matrix"][1][0], "TP": b["confusion_matrix"][1][1]})
p = m["pytorch"]
rows.append({"Model": "PyTorch MLP (Part 3)", "Accuracy": p["accuracy"], "Precision": p["precision"], "Recall": p["recall"],
             "F1": p["f1"], "ROC-AUC": p["roc_auc"], "Train AUC": p["train_auc"], "Gap AUC": p["gap_auc"],
             "Diagnosis": p["diagnosis"], "TN": p["confusion_matrix"][0][0], "FP": p["confusion_matrix"][0][1],
             "FN": p["confusion_matrix"][1][0], "TP": p["confusion_matrix"][1][1]})
df = pd.DataFrame(rows).set_index("Model")
df.round(4).to_csv("reports/comparison_part2_vs_part3.csv")
print(df.round(4).to_string())

# Grafik 1: kurva loss
h = pd.DataFrame(m["history"])
plt.figure(figsize=(6, 3.8)); plt.plot(h.epoch, h.train_loss, label="Train loss"); plt.plot(h.epoch, h.val_loss, label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("BCE (pos_weight)"); plt.title("Kurva Loss - PyTorch MLP"); plt.legend(); plt.tight_layout()
plt.savefig("reports/evaluation/loss_curve.png", dpi=150); plt.close()

# Grafik 2: confusion matrix PyTorch
import numpy as np
cm = np.array(p["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Hadir", "No-show"]); ax.set_yticks([0, 1], ["Hadir", "No-show"])
ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Confusion Matrix - PyTorch MLP"); plt.tight_layout()
plt.savefig("reports/evaluation/confusion_matrix_pytorch.png", dpi=150); plt.close()

# Grafik 3: perbandingan metrik
ax = df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].T.plot(kind="bar", figsize=(8, 4), rot=0)
ax.set_title("Perbandingan Metrik Data Uji: Part 2 vs Part 3"); ax.set_ylim(0, 1); ax.legend(fontsize=8); plt.tight_layout()
plt.savefig("reports/evaluation/comparison_metrics.png", dpi=150); plt.close()
print("grafik tersimpan")
