"""
Step 5: region-based SHAP for one subject, explaining the "Dementia" output.

Usage:
    python scripts/run_shap.py --subject_id OAS1_0365_MR1 [--npermutations 50]

Needs: trained model (train.py) and that subject's masks (build_masks.py).
"""
import argparse
import os

import pandas as pd
import torch

import common
from atlas_interpretability import (explain_regions, load_lut, load_masks,
                                    plot_region_shap, region_names)

TARGET_CLASS = 1    # index of "Dementia" in common.CLASS_NAMES

parser = argparse.ArgumentParser(description="Region-based SHAP for one subject")
parser.add_argument("--subject_id", required=True)
parser.add_argument("--npermutations", type=int, default=50)
args = parser.parse_args()

# Find the subject in the validation or test split (masks exist for both)
subjects = pd.concat([pd.read_csv(os.path.join(common.OUTPUT_DIR, f"{s}_split.csv"))
                      for s in ["val", "test"]], ignore_index=True)
match = subjects[subjects["ID"] == args.subject_id]
if match.empty:
    raise SystemExit(f"{args.subject_id} is not in the validation or test split")
subject = match.iloc[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = common.load_model(common.MODEL_PATH, device)
image = common.eval_transforms()({"image": subject["mri_path"]})["image"].numpy()  # (1, X, Y, Z)
masks = load_masks(common.masks_path(args.subject_id))
names = region_names(masks.keys(), load_lut(common.LUT_PATH))

print(f"{args.subject_id}: {len(masks)} regions, {args.npermutations} permutations")
table, info = explain_regions(model, image, masks, TARGET_CLASS, names=names,
                              npermutations=args.npermutations, device=device)

true_class = common.CLASS_NAMES[int(subject["dementia_label"])]
print(f"True: {true_class} | P(dementia) full brain: {info['prob_full']:.3f} | "
      f"blanked brain: {info['prob_blank']:.3f}")
print(f"Sum of SHAP values: {info['shap_sum']:.4f} "
      f"(should equal {info['prob_full'] - info['prob_blank']:.4f}, "
      f"error {info['additivity_error']:.2e})")
print(table.head(20).to_string(index=False))

os.makedirs(common.SHAP_DIR, exist_ok=True)
table.to_csv(os.path.join(common.SHAP_DIR, f"{args.subject_id}_region_shap.csv"), index=False)
plot_region_shap(
    table,
    title=f"{args.subject_id} | True: {true_class} | P(dementia) = {info['prob_full']:.3f}",
    save_path=os.path.join(common.SHAP_DIR, f"shap_visualization_{args.subject_id}.png"),
    class_name=common.CLASS_NAMES[TARGET_CLASS],
)
print(f"Saved to {common.SHAP_DIR}")
