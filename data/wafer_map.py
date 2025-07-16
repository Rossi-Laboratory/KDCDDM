# data/wafer_map.py
import torch
from torch.utils.data import Dataset

class WaferMapDataset(Dataset):
    """
    Dataset class for wafer map binary or grayscale data.
    """
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img = ...  # Load from self.data_paths[idx]
        if self.transform:
            img = self.transform(img)
        return img, 0
