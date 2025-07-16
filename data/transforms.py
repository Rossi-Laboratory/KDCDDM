# data/transforms.py
from torchvision import transforms

def get_default_transforms():
    """
    Define common image preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
