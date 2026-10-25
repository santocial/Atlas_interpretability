"""
Step 1: build labels and train/val/test splits.

Label rule: CDR > 0 -> 1 (Dementia); CDR == 0 OR missing -> 0 (Healthy).
Note: in OASIS-1, CDR is missing mostly for younger participants (under 60),
who were not clinically rated. They are kept and treated as Healthy.
"""
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

import common

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

os.makedirs(common.OUTPUT_DIR, exist_ok=True)

# Labels
df = pd.read_excel(common.DEMOGRAPHICS_FILE)
df["dementia_label"] = df["CDR"].apply(lambda cdr: 1 if pd.notna(cdr) and cdr > 0 else 0)
df["cdr_missing"] = df["CDR"].isna()
df = df[df["Age"].notna()].copy()

# Keep subjects that have the FreeSurfer brain volume
df["mri_path"] = df["ID"].apply(lambda sid: common.subject_file(sid, common.MRI_FILENAME))
df = df[df["mri_path"].apply(os.path.exists)].copy()
if df.empty:
    raise ValueError(f"No MRI files found under {common.OASIS_DATA_PATH}")

print(f"Subjects: {len(df)} | Healthy: {(df.dementia_label == 0).sum()} "
      f"(of which CDR missing: {df.cdr_missing.sum()}) | "
      f"Dementia: {(df.dementia_label == 1).sum()}")

# Stratified splits: same Healthy/Dementia ratio in each split
train_df, temp_df = train_test_split(
    df, test_size=VAL_RATIO + TEST_RATIO,
    stratify=df["dementia_label"], random_state=common.RANDOM_SEED)
val_df, test_df = train_test_split(
    temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=temp_df["dementia_label"], random_state=common.RANDOM_SEED)

summary = {"label_rule": "CDR > 0 -> Dementia; CDR == 0 or missing -> Healthy"}
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split.to_csv(os.path.join(common.OUTPUT_DIR, f"{name}_split.csv"), index=False)
    summary[name] = {
        "n": len(split),
        "healthy": int((split.dementia_label == 0).sum()),
        "dementia": int((split.dementia_label == 1).sum()),
        "cdr_missing": int(split.cdr_missing.sum()),
    }
    print(f"{name:5s}: {summary[name]}")

with open(os.path.join(common.OUTPUT_DIR, "data_config.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved splits to {common.OUTPUT_DIR}")
