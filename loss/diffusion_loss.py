# loss/diffusion_loss.py
import torch
import torch.nn.functional as F

def diffusion_loss(model, x, t, noise=None):
    """
    Standard diffusion loss (L2 denoising loss) for a diffusion model.

    Args:
        model (nn.Module): The denoising model (e.g., U-Net).
        x (Tensor): Clean input image.
        t (Tensor): Time step tensor.
        noise (Tensor): Optional Gaussian noise; sampled if None.

    Returns:
        Tensor: MSE loss between predicted and actual noise.
    """
    if noise is None:
        noise = torch.randn_like(x)
    x_t = model.q_sample(x, t, noise)
    pred_noise = model.eps_model(x_t, t)
    return F.mse_loss(pred_noise, noise)
