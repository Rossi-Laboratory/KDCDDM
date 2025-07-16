# utils/ema.py
import copy

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    """
    def __init__(self, model, decay):
        self.shadow = copy.deepcopy(model)
        self.decay = decay
        self.shadow.eval()

    def update(self, model):
        with torch.no_grad():
            for shadow_param, param in zip(self.shadow.parameters(), model.parameters()):
                shadow_param.data = self.decay * shadow_param.data + (1.0 - self.decay) * param.data
