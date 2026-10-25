"""Atlas regions: turn a label map into per-region masks, save/load them, look up names."""
import os

import numpy as np


def masks_from_label_map(label_map, min_voxels=10, background=0):
    """
    Split an integer label map (already aligned with the model input) into
    {"region_<label>": uint8 0/1 mask}. Background and regions with
    <= min_voxels voxels are skipped.
    """
    label_map = np.asarray(label_map).astype(int)
    masks = {}
    for label in np.unique(label_map):
        if label == background:
            continue
        mask = (label_map == label).astype(np.uint8)
        if mask.sum() > min_voxels:
            masks[f"region_{label}"] = mask
    return masks


def save_masks(masks, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **masks)


def load_masks(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_lut(lut_path):
    """
    Read a FreeSurfer-style lookup table ("<number> <name> R G B A" per line)
    into {region number: region name}.
    """
    lut = {}
    with open(lut_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                lut[int(parts[0])] = parts[1]
    return lut


def region_names(region_keys, lut):
    """{"region_17": "Left-Hippocampus", ...}; unknown labels keep their key."""
    names = {}
    for key in region_keys:
        try:
            names[key] = lut.get(int(key.split("_")[1]), key)
        except (IndexError, ValueError):
            names[key] = key
    return names
