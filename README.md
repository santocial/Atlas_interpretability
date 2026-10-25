# atlas_interpretability

Region-based SHAP explanations for image classifiers. Each atlas region (e.g. a
FreeSurfer label) is one feature: SHAP blanks out regions, re-runs the model, and
measures how much each region moves the probability of the class you choose.

Works with any PyTorch classifier that returns logits `(batch, n_classes)`, for 2D or 3D images.

The `scripts/` folder is the dementia classification project (OASIS-1, SFCN) that uses it.

## Install

```
pip install -e .                       # the package
pip install -r scripts/requirements.txt  # extra packages for the scripts
```

## The package

| File | What it does |
|---|---|
| `shap_regions.py` | `explain_regions()`: one SHAP value per region |
| `atlas.py` | Label map -> region masks, save/load masks, region-name lookup |
| `plotting.py` | SHAP bar plot + histogram |

```python
from atlas_interpretability import masks_from_label_map, explain_regions

masks = masks_from_label_map(label_map)        # label map aligned with the model input
table, info = explain_regions(model, image, masks, target_class=1)
```

`image` is `(channels, *spatial)`, preprocessed exactly as for the model; `label_map` is `(*spatial)`.

**How "removed" works:** a removed region's voxels are set to `fill_value` (default 0). The
background is a single fully blanked image, so the SHAP values add up to
P(full image) - P(blanked image). `info["additivity_error"]` reports how close they are.

## The scripts (dementia classification)

Edit the paths at the top of `scripts/common.py`, then:

```
python scripts/prepare_data.py     # labels + train/val/test splits
python scripts/train.py            # trains SFCN, saves best_model_sfcn.pth
python scripts/evaluate.py         # metrics, predictions CSV, ROC plot
python scripts/build_masks.py      # region masks for val + test subjects
python scripts/run_shap.py --subject_id OAS1_0365_MR1
```

`common.py` holds the paths, the SFCN model and the preprocessing shared by all scripts.

### Notes and decisions

- **Labels:** CDR > 0 is Dementia. CDR = 0 *or missing* is Healthy. In OASIS-1, missing CDR is
  mostly younger participants who were not clinically rated. A `cdr_missing` column is kept.
- **SHAP background:** only the fully blanked brain. (The earlier version also used the full
  brain as a second background, which halved all values.)
- **Augmentation:** random shift of up to 2 voxels + left-right flip (p = 0.5). The earlier
  version used a 90-degree rotation by mistake.
- **Mask alignment:** masks are cropped with the MRI's crop box, so they line up voxel-for-voxel
  with the model input.
