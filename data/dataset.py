import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch  # Add this import

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.transform = transform
        self.image_paths = []

        for root in root_dirs:
            for subdir, _, files in os.walk(root):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(subdir, file))

        print(f"[DEBUG] Loaded {len(self.image_paths)} images from {root_dirs}")
        if len(self.image_paths) == 0:
            print("[WARNING] No images found! Check directory paths or extensions.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, labels_json, transform=None):
        self.transform = transform
        self.samples = []

        # Load the label mapping (folder -> label_name)
        with open(labels_json, 'r') as f:
            self.label_map = json.load(f)

        # Build set of label names and create mapping to class indices
        all_labels = sorted(set(self.label_map.values()))
        self.class_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        for class_folder, label_str in self.label_map.items():
            folder_path = os.path.join(root_dir, class_folder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, fname)
                    label_idx = self.class_to_idx[label_str]  # Convert string to numeric label
                    self.samples.append((img_path, label_idx))

        print(f"[DEBUG] Loaded {len(self.samples)} labeled images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label_idx)  # ✅ Return tensor label



unlabeled_paths = [
    "data/train.X1",
    "data/train.X2",
    "data/train.X3",
    "data/train.X4"
]

# Example usage (without transforms)
unlabeled_dataset = UnlabeledImageDataset(root_dirs=unlabeled_paths)
