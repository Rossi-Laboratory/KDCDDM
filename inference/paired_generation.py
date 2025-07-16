# inference/paired_generation.py
import torch
from models.cdldm import CDLDM

def generate_paired_images(model: CDLDM, x_defect):
    """
    Generate paired (defect, clean) image from defect input using CDLDM.
    Args:
        model (CDLDM): Trained CDLDM model.
        x_defect (Tensor): Input defect image.
    Returns:
        Tuple[Tensor, Tensor]: (Original defect, Synthesized circuit)
    """
    model.eval()
    with torch.no_grad():
        x_circuit = model.generate_circuit_from_defect(x_defect.cuda())
        return x_defect, x_circuit
