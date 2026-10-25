"""
Step 2: train SFCN (Healthy vs Dementia).

Same setup as before: weighted cross-entropy, SGD (lr 0.01, momentum 0.9,
weight decay 0.001), lr x0.3 every 30 epochs, 130 epochs, keep the checkpoint
with the best validation balanced accuracy.
Change: augmentation is now shift + left-right flip (was a 90-degree rotation).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.data import PersistentDataset, list_data_collate
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

import common

NUM_EPOCHS = 130
BATCH_SIZE = 8
LEARNING_RATE = 0.01
LR_DECAY = 0.3
LR_DECAY_EPOCHS = 30
WEIGHT_DECAY = 0.001
DROPOUT_RATE = 0.5
SAVE_EVERY = 10

torch.manual_seed(common.RANDOM_SEED)
np.random.seed(common.RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(common.MODEL_DIR, exist_ok=True)


def make_loader(split, transform, shuffle):
    df = pd.read_csv(os.path.join(common.OUTPUT_DIR, f"{split}_split.csv"))
    files = [{"image": r.mri_path, "label": r.dementia_label} for r in df.itertuples()]
    ds = PersistentDataset(files, transform=transform,
                           cache_dir=os.path.join(common.CACHE_DIR, split))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4,
                        pin_memory=True, collate_fn=list_data_collate)
    return df, loader


train_df, train_loader = make_loader("train", common.train_transforms(), shuffle=True)
val_df, val_loader = make_loader("val", common.eval_transforms(), shuffle=False)

# Class weights: total / (2 * class count), so the rarer class counts more
counts = train_df["dementia_label"].value_counts().sort_index().values
class_weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float32)
print(f"Device: {device} | class weights (Healthy, Dementia): {class_weights.tolist()}")

model = common.SFCN(num_classes=common.NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                            momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_DECAY_EPOCHS, LR_DECAY)


def run_epoch(loader, train):
    """One pass over the data. Returns mean loss, balanced accuracy, ROC-AUC."""
    model.train(train)
    total_loss, labels_all, probs_all = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).long()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            labels_all.append(y.cpu().numpy())
            probs_all.append(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
    labels, probs = np.concatenate(labels_all), np.concatenate(probs_all)
    bal_acc = balanced_accuracy_score(labels, probs >= 0.5)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan")
    return total_loss / len(loader), bal_acc, auc


def save_checkpoint(path, epoch, val_acc, history):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "history": history,
        "num_classes": common.NUM_CLASSES,
        "class_names": common.CLASS_NAMES,
        "class_weights": class_weights.tolist(),
    }, path)


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
best_acc, best_epoch = 0.0, 0

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc, _ = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_auc = run_epoch(val_loader, train=False)
    scheduler.step()

    for key, value in zip(history, [tr_loss, tr_acc, va_loss, va_acc, va_auc]):
        history[key].append(value)
    print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} bal.acc {tr_acc:.3f} | "
          f"val loss {va_loss:.4f} bal.acc {va_acc:.3f} AUC {va_auc:.3f}")

    if va_acc > best_acc:
        best_acc, best_epoch = va_acc, epoch
        save_checkpoint(common.MODEL_PATH, epoch, va_acc, history)
        print("  -> new best model saved")
    if epoch % SAVE_EVERY == 0:
        save_checkpoint(os.path.join(common.MODEL_DIR, f"checkpoint_sfcn_epoch_{epoch}.pth"),
                        epoch, va_acc, history)

pd.DataFrame(history, index=range(1, NUM_EPOCHS + 1)).rename_axis("epoch").to_csv(
    os.path.join(common.MODEL_DIR, "training_history_sfcn.csv"))
with open(os.path.join(common.MODEL_DIR, "training_summary_sfcn.json"), "w") as f:
    json.dump({"best_epoch": best_epoch, "best_val_bal_acc": best_acc,
               "augmentation": "shift <=2 voxels + left-right flip (p=0.5)"}, f, indent=2)
print(f"Done. Best val balanced accuracy {best_acc:.4f} at epoch {best_epoch}")
