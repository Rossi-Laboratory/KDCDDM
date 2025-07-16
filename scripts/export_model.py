# scripts/export_model.py
import torch

def export_model(model, path="exported_model.pth"):
    """
    Save model weights to disk.
    """
    torch.save(model.state_dict(), path)
    print(f"Model exported to {path}")
