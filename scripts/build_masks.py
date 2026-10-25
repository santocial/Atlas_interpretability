"""
Step 4: build FreeSurfer region masks for every validation and test subject.

Preprocessing (crop + resize) is done here with MONAI, so the masks line up
with the model input; atlas_interpretability then splits the label map into
one 0/1 mask per region. Output: <MASKS_DIR>/<ID>_masks.npz
"""
import os

import pandas as pd
from tqdm import tqdm

import common
from atlas_interpretability import masks_from_label_map, save_masks

subjects = pd.concat([pd.read_csv(os.path.join(common.OUTPUT_DIR, f"{s}_split.csv"))
                      for s in ["val", "test"]], ignore_index=True)
transform = common.image_and_segmentation_transforms()

failed = []
for row in tqdm(subjects.itertuples(), total=len(subjects)):
    seg_path = common.subject_file(row.ID, common.SEG_FILENAME)
    try:
        if not os.path.exists(seg_path):
            raise FileNotFoundError(seg_path)
        data = transform({"image": row.mri_path, "seg": seg_path})
        label_map = data["seg"][0].numpy()          # drop channel dim
        save_masks(masks_from_label_map(label_map, min_voxels=10),
                   common.masks_path(row.ID))
    except Exception as e:     # keep going; report failures at the end
        failed.append({"ID": row.ID, "reason": str(e)})

print(f"Done: {len(subjects) - len(failed)}/{len(subjects)} subjects -> {common.MASKS_DIR}")
if failed:
    path = os.path.join(common.OUTPUT_DIR, "segmentation_processing_failed.csv")
    pd.DataFrame(failed).to_csv(path, index=False)
    print(f"{len(failed)} failed, see {path}")
