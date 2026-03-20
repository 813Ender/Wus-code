import os
import pydicom
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from utils.helpers import collate_fn


class MultiTaskDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform

        for case in sorted(os.listdir(root_dir)):
            case_path = os.path.join(root_dir, case)
            if not os.path.isdir(case_path):
                continue

            dwi_dir = os.path.join(case_path, "DWI")
            flair_dir = os.path.join(case_path, "FLAIR")

            if not (os.path.isdir(dwi_dir) and os.path.isdir(flair_dir)):
                print(f"Skipping case missing DWI/FLAIR: {case}")
                continue

            try:
                number_str = case.split("min")[0].split("_")[-1]
                number = int(number_str)
                time_label = 1 if number >= 270 else 0
            except (ValueError, IndexError):
                print(f"Skipping case with invalid name (cannot parse time): {case}")
                continue

            dwi_files = sorted([f for f in os.listdir(dwi_dir) if f.endswith(".dcm")])
            flair_files = sorted([f for f in os.listdir(flair_dir) if f.endswith(".dcm")])
            common_files = set(dwi_files).intersection(set(flair_files))
            if not common_files:
                print(f"Case {case} has no matching DWI/FLAIR slices")
                continue

            for fname in sorted(common_files):
                dwi_path = os.path.join(dwi_dir, fname)
                flair_path = os.path.join(flair_dir, fname)
                relative_path = os.path.join(case, "DWI", fname)
                lesion_label = 1 if "x" in fname.lower() else 0
                self.samples.append(
                    (dwi_path, flair_path, lesion_label, time_label, relative_path)
                )

        if not self.samples:
            raise ValueError(f"No valid samples found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Use SampleDataset for __getitem__")


class SampleDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dwi_path, flair_path, lesion_label, time_label, relative_path = self.samples[
            idx
        ]

        try:
            dwi_ds = pydicom.dcmread(dwi_path)
            flair_ds = pydicom.dcmread(flair_path)
            dwi = dwi_ds.pixel_array.astype(np.float32)
            flair = flair_ds.pixel_array.astype(np.float32)

            if len(dwi.shape) > 2:
                dwi = dwi[0]
            if len(flair.shape) > 2:
                flair = flair[0]

            dwi = (dwi - np.min(dwi)) / (np.ptp(dwi) + 1e-5)
            flair = (flair - np.min(flair)) / (np.ptp(flair) + 1e-5)

            dwi_img = Image.fromarray((dwi * 255).astype(np.uint8)).convert("L")
            flair_img = Image.fromarray((flair * 255).astype(np.uint8)).convert("L")

            dwi_img = dwi_img.resize((224, 224), resample=Image.BILINEAR)
            flair_img = flair_img.resize((224, 224), resample=Image.BILINEAR)

            dwi_tensor = torch.from_numpy(np.array(dwi_img)).float() / 255.0
            flair_tensor = torch.from_numpy(np.array(flair_img)).float() / 255.0
            diff_tensor = torch.clamp(dwi_tensor - flair_tensor, min=0)

            img_tensor = torch.stack([dwi_tensor, flair_tensor, diff_tensor], dim=0)
            img_tensor = (img_tensor - 0.5) / 0.5

            if self.transform:
                img_tensor = self.transform(img_tensor)

            return (
                img_tensor,
                torch.tensor(lesion_label, dtype=torch.long),
                torch.tensor(time_label, dtype=torch.long),
                relative_path,
            )

        except Exception as e:
            print(f"Error loading sample {relative_path}: {e}")
            return None


def build_loaders(train_samples, val_samples, batch_size=16, num_workers=4):

    train_dataset = SampleDataset(train_samples)
    val_dataset = SampleDataset(val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
