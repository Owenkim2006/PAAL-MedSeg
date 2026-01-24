"""
Preprocess BraTS 2023 dataset for PAAL-MedSeg:
Converts 3D MRI volumes (4 modalities + segmentation) into 2D HDF5 slices.

Each sample will be saved as:
  dataset/BraTS2023/2d_data-norm/<caseID>_<sliceIdx>.hdf5

Each HDF5 file contains:
  - image: (4, H, W) float32 array normalized to [0,1]
  - label: (H, W) segmentation mask
"""

import os
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from skimage.transform import resize


def load_nifti_volume(nii_path):
    """Load NIfTI file and return numpy array."""
    img = nib.load(nii_path)
    return img.get_fdata()


def normalize_image(image):
    """Normalize image channels to [0,1]."""
    for c in range(image.shape[0]):
        ch = image[c]
        min_val, max_val = ch.min(), ch.max()
        if max_val - min_val > 0:
            image[c] = (ch - min_val) / (max_val - min_val)
    return image


def extract_2d_slices(image_4d, label_3d, case_id, output_dir, target_shape=(256, 256)):
    """Extract and save 2D axial slices as HDF5 files."""
    depth = image_4d.shape[-1]
    saved_files = []

    for idx in range(depth):
        img_slice = image_4d[:, :, :, idx]
        lbl_slice = label_3d[:, :, idx]

        if lbl_slice.max() == 0:
            continue  # skip empty slices

        # Resize slices
        resized_img = np.zeros((4,) + target_shape, dtype=np.float32)
        for c in range(4):
            resized_img[c] = resize(img_slice[c], target_shape, order=1, anti_aliasing=True)

        resized_lbl = resize(lbl_slice, target_shape, order=0, preserve_range=True, anti_aliasing=False)
        resized_lbl = (resized_lbl > 0.5).astype(np.float32)

        # Normalize
        resized_img = normalize_image(resized_img)

        # Save to HDF5
        filename = f"{case_id}_{idx}.hdf5"
        filepath = os.path.join(output_dir, filename)
        with h5py.File(filepath, "w") as hf:
            hf.create_dataset("image", data=resized_img.astype(np.float32))
            hf.create_dataset("label", data=resized_lbl.astype(np.float32))

        saved_files.append(filepath)
    return saved_files


def create_split_json(all_files, output_path, num_folds=5, seed=42):
    """Create k-fold split JSON."""
    np.random.seed(seed)
    indices = np.arange(len(all_files))
    np.random.shuffle(indices)

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

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits JSON to {output_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    nii_root_dir = project_root / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    output_dir = project_root / "dataset/BraTS2023/2d_data-norm"
    split_json_path = project_root / "dataset/BraTS2023/split.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all subject directories
    subject_dirs = sorted([p for p in nii_root_dir.iterdir() if p.is_dir()])
    print(f"Found {len(subject_dirs)} subjects.")

    all_hdf5 = []

    # Recursively search for a file in a folder (legacy helper)
    def find_file_recursive(folder: Path, filename: str):
        """Recursively search for a file inside folder."""
        for path in folder.rglob(filename):
            return path
        return None

    def identify_modality_files(folder: Path, case_id: str):
        """Identify modality files by common substrings and return a mapping.

        This is more robust than exact filenames because datasets may use
        different suffixes (e.g., `-t1c.nii.gz`, `-t2f.nii.gz`, etc.).
        """
        mapping = {"flair": None, "t1": None, "t1ce": None, "t2": None, "seg": None}

        # gather candidate nii files (accept .nii and .nii.gz)
        candidates = [p for p in folder.iterdir() if p.is_file() and (str(p).lower().endswith('.nii') or str(p).lower().endswith('.nii.gz'))]

        for p in candidates:
            name = p.name.lower()
            # segmentation
            if 'seg' in name or 'mask' in name or 'label' in name:
                mapping['seg'] = p
                continue

            # T1 contrast-enhanced
            if 't1c' in name or 't1ce' in name or '-t1ce' in name:
                mapping['t1ce'] = p
                continue

            # T1 native (sometimes named t1n or t1)
            if 't1n' in name or (('t1' in name) and ('t1c' not in name and 't1ce' not in name)):
                mapping['t1'] = p
                continue

            # FLAIR (sometimes named flair, t2f, or t2flair)
            if 'flair' in name or 't2f' in name or 't2flair' in name:
                mapping['flair'] = p
                continue

            # T2 (t2w or t2)
            if 't2w' in name or (('t2' in name) and ('t2f' not in name and 't2flair' not in name)):
                mapping['t2'] = p
                continue

        return mapping

    for subj_path in tqdm(subject_dirs, desc="Processing BraTS 2023 volumes"):
        case_id = subj_path.name  # e.g., BraTS-GLI-00000-000

        # Try to identify modality files using flexible patterns (handles .nii.gz and different suffixes)
        mod_map = identify_modality_files(subj_path, case_id)

        flair = mod_map.get('flair')
        t1 = mod_map.get('t1')
        t1ce = mod_map.get('t1ce')
        t2 = mod_map.get('t2')
        seg = mod_map.get('seg')

        missing = [k for k, v in [('flair', flair), ('t1', t1), ('t1ce', t1ce), ('t2', t2), ('seg', seg)] if v is None]
        if missing:
            print(f"Skipping {case_id}: Missing modality files: {missing}")
            print("Found files:")
            for p in sorted(subj_path.iterdir()):
                if p.is_file():
                    print("  ", p.name)
            continue

        try:
            img_stack = np.stack([
                load_nifti_volume(flair),
                load_nifti_volume(t1ce),
                load_nifti_volume(t1),
                load_nifti_volume(t2),
            ], axis=0)
            lbl_data = load_nifti_volume(seg)

            saved = extract_2d_slices(img_stack, lbl_data, case_id, str(output_dir))
            all_hdf5.extend(saved)
        except Exception as e:
            print(f"Error processing {case_id}: {e}")

    print(f"\nTotal 2D slices saved: {len(all_hdf5)}")
    create_split_json(all_hdf5, split_json_path)
    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
