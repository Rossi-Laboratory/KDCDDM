# data/etch_sem.py
import torch
from torch.utils.data import Dataset

class EtchSEMDataset(Dataset):
    """
    Dataset class for SEM etching images.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [...]  # populate with actual file paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = ...  # Load image from self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, 0
