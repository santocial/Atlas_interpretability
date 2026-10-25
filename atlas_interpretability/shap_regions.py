"""
Region-based SHAP: each atlas region is one feature.

A feature value of 1 keeps the region; 0 blanks it (sets its voxels to
fill_value). SHAP compares the model's probability for `target_class` with and
without each region, and splits the difference between the full image and a
fully blanked image across the regions.

Works with any PyTorch classifier that returns logits of shape (batch, n_classes),
for 2D or 3D images.
"""
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F


def make_predict_fn(model, image, region_masks, region_keys, target_class,
                    fill_value=0.0, device="cpu"):
    """
    Build f(presence) -> probability of target_class, where presence has shape
    (n_samples, n_regions) with 1 = keep region, 0 = blank it.
    `image` has shape (channels, *spatial); each mask has shape (*spatial).
    """
    def predict(presence):
        presence = np.atleast_2d(presence)
        probs = np.zeros(len(presence))
        for i, row in enumerate(presence):
            masked = image.copy()
            for keep, key in zip(row, region_keys):
                if keep == 0:
                    masked[:, region_masks[key] == 1] = fill_value
            x = torch.from_numpy(masked).float().unsqueeze(0).to(device)
            with torch.no_grad():
                probs[i] = F.softmax(model(x), dim=1)[0, target_class].item()
        return probs
    return predict


def explain_regions(model, image, region_masks, target_class, names=None,
                    npermutations=50, fill_value=0.0, device="cpu"):
    """
    Compute one SHAP value per region for one image.

    Background = all regions blanked (a single row of zeros), so the SHAP values
    add up to P(full image) - P(blanked image).

    Args:
      model:         PyTorch model in eval mode, returns logits
      image:         numpy array (channels, *spatial), preprocessed as for the model
      region_masks:  {key: 0/1 array of shape (*spatial)}
      target_class:  index of the output class to explain
      names:         optional {key: readable name}
    Returns (table, info):
      table: DataFrame sorted by |SHAP| with region_key, region_name,
             shap_value, abs_shap_value, n_voxels
      info:  dict with prob_full, prob_blank, shap_sum, additivity_error
    """
    region_keys = sorted(region_masks)
    n = len(region_keys)
    names = names or {}
    predict = make_predict_fn(model, image, region_masks, region_keys,
                              target_class, fill_value, device)

    explainer = shap.PermutationExplainer(predict, np.zeros((1, n)))
    shap_values = np.asarray(
        explainer.shap_values(np.ones((1, n)), npermutations=npermutations))[0]

    # Check: SHAP values should add up to P(full) - P(blanked)
    prob_full, prob_blank = predict(np.vstack([np.ones(n), np.zeros(n)]))
    info = {
        "prob_full": float(prob_full),
        "prob_blank": float(prob_blank),
        "shap_sum": float(shap_values.sum()),
        "additivity_error": float(abs(shap_values.sum() - (prob_full - prob_blank))),
    }

    table = pd.DataFrame({
        "region_key": region_keys,
        "region_name": [names.get(k, k) for k in region_keys],
        "shap_value": shap_values,
        "abs_shap_value": np.abs(shap_values),
        "n_voxels": [int(region_masks[k].sum()) for k in region_keys],
    }).sort_values("abs_shap_value", ascending=False, ignore_index=True)

    return table, info
