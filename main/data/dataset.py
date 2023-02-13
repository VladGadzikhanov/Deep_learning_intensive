import albumentations
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, targets):
        self.image_paths = image_paths
        self.targets = targets
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(always_apply=True),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        target = self.targets[item]

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.long),
        }
