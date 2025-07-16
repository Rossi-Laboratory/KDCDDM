# inference/defect_detection.py
import torch
from models.icddm import ICDDM

def detect_defect(model: ICDDM, x_circuit):
    """
    Uses ICDDM to infer potential defects from a clean circuit image.
    Args:
        model (ICDDM): Trained ICDDM model.
        x_circuit (Tensor): Input clean circuit image.
    Returns:
        Tensor: Synthesized defect prediction.
    """
    model.eval()
    with torch.no_grad():
        return model.generate_defect_from_circuit(x_circuit.cuda())
