# models/cdldm.py
import torch
import torch.nn as nn
from models.diffusion import Diffusion, CosineSchedule

class CDLDM(nn.Module):
    """
    Cross Domain Latent Diffusion Model (CDLDM)
    Performs diffusion in latent space using pre-trained VAE encoder/decoder.
    """
    def __init__(self, encoder, decoder, unet_defect, unet_circuit, T=1000):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unet_defect = unet_defect
        self.unet_circuit = unet_circuit
        self.schedule = CosineSchedule(T)
        self.T = T
        self.diff_defect = Diffusion(T, self.unet_defect, self.schedule)
        self.diff_circuit = Diffusion(T, self.unet_circuit, self.schedule)

    def training_step(self, x_defect, x_circuit, t):
        """
        Training loss for CDLDM in latent space.

        Args:
            x_defect (Tensor): Defect image input.
            x_circuit (Tensor): Circuit map input.
            t (Tensor): Time step tensor.

        Returns:
            Tensor: Total loss from both latent diffusion directions.
        """
        z_defect_mu, _ = self.encoder(x_defect)
        z_circuit_mu, _ = self.encoder(x_circuit)

        noise_defect = torch.randn_like(z_defect_mu)
        noise_circuit = torch.randn_like(z_circuit_mu)

        zt_defect = self.diff_defect.q_sample(z_defect_mu, t, noise_defect)
        zt_circuit = self.diff_circuit.q_sample(z_circuit_mu, t, noise_circuit)

        pred_defect = self.unet_defect(zt_defect, t)
        pred_circuit = self.unet_circuit(zt_circuit, t)

        loss_d = nn.functional.mse_loss(pred_defect, noise_defect)
        loss_c = nn.functional.mse_loss(pred_circuit, noise_circuit)
        return loss_d + loss_c

    @torch.no_grad()
    def generate_circuit_from_defect(self, x_defect):
        """
        Generate clean circuit image from defect image using latent reverse diffusion.

        Args:
            x_defect (Tensor): Defect image input.

        Returns:
            Tensor: Synthesized circuit image.
        """
        z = torch.randn_like(self.encoder(x_defect)[0])
        for t in reversed(range(self.T)):
            t_tensor = torch.full((z.shape[0],), t, dtype=torch.long, device=z.device)
            eps = self.unet_circuit(z, t_tensor)
            alpha = self.schedule.alphas[t]
            z = (z - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        return self.decoder(z)

    @torch.no_grad()
    def generate_defect_from_circuit(self, x_circuit):
        """
        Generate defect image from clean circuit map using latent reverse diffusion.

        Args:
            x_circuit (Tensor): Circuit image input.

        Returns:
            Tensor: Synthesized defect image.
        """
        z = torch.randn_like(self.encoder(x_circuit)[0])
        for t in reversed(range(self.T)):
            t_tensor = torch.full((z.shape[0],), t, dtype=torch.long, device=z.device)
            eps = self.unet_defect(z, t_tensor)
            alpha = self.schedule.alphas[t]
            z = (z - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        return self.decoder(z)
