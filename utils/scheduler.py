# utils/scheduler.py
import math

class CosineSchedule:
    """
    Cosine noise schedule for diffusion timesteps.
    Computes alphas and betas based on cosine annealing.
    """
    def __init__(self, T):
        self.T = T
        self.alphas = [math.cos((t / T) * math.pi / 2) ** 2 for t in range(T)]
