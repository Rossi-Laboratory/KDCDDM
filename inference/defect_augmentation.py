# inference/defect_augmentation.py
import torch
from models.cdldm import CDLDM

def augment_defect(model: CDLDM, x_defect):
    """
    Performs latent space augmentation on defect images via CDLDM.
    Args:
        model (CDLDM): Trained CDLDM model.
        x_defect (Tensor): Input defect image.
    Returns:
        Tensor: Augmented synthetic defects.
    """
    model.eval()
    with torch.no_grad():
        return model.generate_defect_from_circuit(x_defect.cuda())
