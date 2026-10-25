"""
Shared code for the dementia classification scripts: paths, settings, the SFCN
model and MONAI preprocessing. Edit the paths below to match your system.

OUTPUT_DIR is the same folder the original scripts used, so existing splits,
checkpoints and caches are found without moving anything.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    Spacingd,
)

# ============================================================================
# Paths (edit these)
# ============================================================================
BASE_PATH = "/data/pt_02271/Analysis/Nako_BrainAge"
OASIS_DATA_PATH = f"{BASE_PATH}/Datasets/OASIS"
DEMOGRAPHICS_FILE = f"{OASIS_DATA_PATH}/oasis_cross-sectional-5708aa0a98d82080.xlsx"
OUTPUT_DIR = (f"{BASE_PATH}/NotNako/OASIS_analysis/region_based_explanations/"
              "dementia_classification/derivatives")

CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models_sfcn")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_sfcn.pth")
MASKS_DIR = os.path.join(OUTPUT_DIR, "segmentation_masks")
SHAP_DIR = os.path.join(OUTPUT_DIR, "region_shap_results")
LUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreeSurferColorLUT.txt")

MRI_FILENAME = "brain.finalsurfs.mgz"
SEG_FILENAME = "aparc+aseg.mgz"

# ============================================================================
# Settings
# ============================================================================
RANDOM_SEED = 42
CLASS_NAMES = ["Healthy", "Dementia"]   # label 0, label 1
NUM_CLASSES = 2
IMAGE_SIZE = (96, 96, 96)
VOXEL_SPACING = (1.0, 1.0, 1.0)


def subject_file(subject_id, filename):
    """Path to a FreeSurfer file: <OASIS_DATA_PATH>/<ID>/mri/<filename>."""
    return os.path.join(OASIS_DATA_PATH, subject_id, "mri", filename)


def masks_path(subject_id):
    return os.path.join(MASKS_DIR, f"{subject_id}_masks.npz")


# ============================================================================
# Model: SFCN (Peng et al. 2021), 2-class output
# ============================================================================
class SFCN(nn.Module):
    """
    Blocks 1-5: Conv3d(3x3x3) -> BatchNorm -> MaxPool -> ReLU
    Block 6:    Conv3d(1x1x1) -> BatchNorm -> ReLU
    Block 7:    AvgPool -> Dropout -> Conv3d(1x1x1)
    Returns raw scores (logits). Layer names match the original scripts,
    so old checkpoints still load.
    """

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.pool5 = nn.MaxPool3d(2)

        self.conv6 = nn.Conv3d(256, 64, kernel_size=1)
        self.bn6 = nn.BatchNorm3d(64)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.conv7 = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))
        x = F.relu(self.pool4(self.bn4(self.conv4(x))))
        x = F.relu(self.pool5(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(self.dropout(self.avgpool(x)))
        return x.view(x.size(0), -1)


def load_model(model_path=MODEL_PATH, device="cpu"):
    """Load a checkpoint saved by train.py, in eval mode."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SFCN(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ============================================================================
# Preprocessing: load -> RAS -> 1 mm voxels -> scale to [0, 1] -> crop -> resize
# ============================================================================
# Crop bookkeeping keys that break DataLoader batching
_CROP_KEYS = ["foreground_start_coord", "foreground_end_coord",
              "foreground_original_shape", "foreground_cropped_shape"]


def _image_steps():
    return [
        LoadImaged(keys=["image"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=VOXEL_SPACING, mode="bilinear"),
        ScaleIntensityd(keys=["image"]),
        CropForegroundd(keys=["image"], source_key="image", margin=10),
        Resized(keys=["image"], spatial_size=IMAGE_SIZE),
    ]


def eval_transforms():
    """Deterministic preprocessing for validation, test and SHAP."""
    return Compose(_image_steps() + [DeleteItemsd(keys=_CROP_KEYS)])


def train_transforms():
    """Preprocessing plus augmentation: shift up to 2 voxels, left-right flip (50%)."""
    return Compose(_image_steps() + [
        RandAffined(keys=["image"], prob=1.0, translate_range=(2, 2, 2),
                    padding_mode="zeros"),
        # After RAS orientation, spatial axis 0 is left-right
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        DeleteItemsd(keys=_CROP_KEYS),
    ])


def image_and_segmentation_transforms():
    """
    Preprocess an MRI and its FreeSurfer segmentation together. The crop box is
    taken from the MRI and applied to both, so the region masks line up
    voxel-for-voxel with the model input. Nearest-neighbour keeps labels whole.
    """
    keys = ["image", "seg"]
    return Compose([
        LoadImaged(keys=keys, reader="NibabelReader"),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        Spacingd(keys=keys, pixdim=VOXEL_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        CropForegroundd(keys=keys, source_key="image", margin=10),
        Resized(keys=keys, spatial_size=IMAGE_SIZE, mode=("area", "nearest")),
        DeleteItemsd(keys=_CROP_KEYS),
    ])
