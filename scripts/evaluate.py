"""
Step 3: evaluate the best model on the validation and test sets.
Saves per-subject predictions (CSV), metrics (JSON) and ROC curves (PNG).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import PersistentDataset, list_data_collate
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

import common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = common.load_model(common.MODEL_PATH, device)


def predict_split(split):
    """Return the split's DataFrame plus true labels and dementia probabilities."""
    df = pd.read_csv(os.path.join(common.OUTPUT_DIR, f"{split}_split.csv"))
    files = [{"image": r.mri_path, "label": r.dementia_label} for r in df.itertuples()]
    ds = PersistentDataset(files, transform=common.eval_transforms(),
                           cache_dir=os.path.join(common.CACHE_DIR, split))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=list_data_collate)
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.append(batch["label"].numpy())
    return df, np.concatenate(labels).astype(int), np.concatenate(probs)


def metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall_sensitivity": recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


results, all_rows = {}, []
fig, ax = plt.subplots(figsize=(6, 6))

for split in ["val", "test"]:
    df, labels, probs = predict_split(split)
    m = metrics(labels, probs)
    results[split] = {k: (float(v) if not isinstance(v, dict) else v) for k, v in m.items()}
    print(f"\n{split.upper()}: " + ", ".join(
        f"{k}={v:.3f}" for k, v in m.items() if not isinstance(v, dict)))
    print(f"  confusion matrix: {m['confusion_matrix']}")

    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, label=f"{split} (AUC = {m['roc_auc']:.3f})")

    all_rows.append(pd.DataFrame({
        "ID": df["ID"], "set": split, "true_label": labels,
        "predicted_label": (probs >= 0.5).astype(int),
        "prob_dementia": probs, "cdr_missing": df["CDR"].isna(),
    }))

ax.plot([0, 1], [0, 1], "k--", label="chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("SFCN - ROC curves")
ax.legend(loc="lower right")
fig.savefig(os.path.join(common.MODEL_DIR, "sfcn_roc.png"), dpi=150, bbox_inches="tight")

pd.concat(all_rows).to_csv(os.path.join(common.MODEL_DIR, "sfcn_predictions.csv"), index=False)
with open(os.path.join(common.MODEL_DIR, "sfcn_evaluation_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved predictions, metrics and ROC plot to {common.MODEL_DIR}")
