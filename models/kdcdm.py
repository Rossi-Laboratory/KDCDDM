
import torch
import torch.nn as nn
import torch.nn.functional as F

class KDCDDM(nn.Module):
    """
    Knowledge Distillation Cross Domain Diffusion Model (KDCDDM)
    Trains a fast generator (student) to mimic slow CDLDM (teacher) output.
    """
    def __init__(self, teacher_model, generator_G, discriminator_D, lambda_l1=10.0):
        super().__init__()
        self.teacher = teacher_model
        self.G = generator_G  # student generator
        self.D = discriminator_D  # GAN discriminator
        self.lambda_l1 = lambda_l1

    def generator_loss(self, z_defect, z_teacher):
        """
        Computes generator (student) loss.
        Args:
            z_defect: noisy latent of defect
            z_teacher: latent output of teacher diffusion model
        Returns:
            total_loss: adversarial + L1 loss
        """
        z_fake = self.G(z_defect)
        pred_fake = self.D(z_fake)

        adv_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
        l1_loss = F.l1_loss(z_fake, z_teacher.detach())
        return adv_loss + self.lambda_l1 * l1_loss

    def discriminator_loss(self, z_fake, z_real):
        """
        Computes GAN discriminator loss.
        """
        pred_real = self.D(z_real)
        pred_fake = self.D(z_fake.detach())
        loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (loss_real + loss_fake)

    @torch.no_grad()
    def inference(self, z_defect):
        """
        Inference with student generator.
        """
        return self.G(z_defect)
