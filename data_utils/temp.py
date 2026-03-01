# run this as a one-off script: generate_split.py
import json
import numpy as np
from pathlib import Path

output_dir = Path('./dataset/BraTS2023/2d_data-norm')
split_json_path = Path('./dataset/BraTS2023/split.json')

# Collect all existing hdf5 files
all_files = sorted([str(p) for p in output_dir.glob('*.hdf5')])
print(f"Found {len(all_files)} slices")

# Create 5-fold split (same seed as preprocessor so it's reproducible)
np.random.seed(42)
indices = np.arange(len(all_files))
np.random.shuffle(indices)

num_folds = 5
fold_size = len(all_files) // num_folds
splits = {}

for fold in range(1, num_folds + 1):
    val_start = (fold - 1) * fold_size
    val_end = val_start + fold_size if fold < num_folds else len(all_files)
    val_idx = indices[val_start:val_end]
    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

    splits[f"fold{fold}"] = {
        "train_path": [all_files[i] for i in train_idx],
        "val_path": [all_files[i] for i in val_idx],
    }

with open(split_json_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"✅ split.json saved to {split_json_path}")
print(f"Fold sizes — train: {len(splits['fold1']['train_path'])}, val: {len(splits['fold1']['val_path'])}")