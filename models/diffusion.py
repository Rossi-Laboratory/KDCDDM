# models/diffusion.py
import numpy as np

class CosineSchedule:
    """
    Cosine-based beta schedule for diffusion noise levels.
    """
    def __init__(self, T):
        self.T = T
        self.betas = self._make_betas()

    def _make_betas(self):
        steps = np.linspace(0, np.pi / 2, self.T + 1)
        alphas = np.cos(steps) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.tensor(betas, dtype=torch.float32)

class Diffusion:
    """
    Implements forward (q_sample) and reverse (p_sample) diffusion process.
    Args:
        T (int): number of diffusion steps.
        model (nn.Module): denoising model.
        schedule (CosineSchedule): precomputed beta schedule.
    """
    def __init__(self, T, model, schedule: CosineSchedule):
        self.T = T
        self.model = model
        self.betas = schedule.betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: add noise to clean input x0 at step t.
        """
        alpha_cum = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(alpha_cum) * x0 + torch.sqrt(1 - alpha_cum) * noise

    def p_sample(self, xt, t, cond):
        """
        Reverse diffusion: denoise current sample xt at step t using the model.
        """
        eps_pred = self.model(xt, cond)
        alpha_t = self.alphas[t][:, None, None, None]
        return (xt - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
