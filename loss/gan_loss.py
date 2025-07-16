# loss/gan_loss.py
import torch.nn.functional as F

def gan_generator_loss(pred_fake):
    """
    GAN generator loss using binary cross-entropy with real labels.

    Args:
        pred_fake (Tensor): Discriminator prediction on fake samples.

    Returns:
        Tensor: BCE loss.
    """
    return F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))

def gan_discriminator_loss(pred_real, pred_fake):
    """
    GAN discriminator loss for distinguishing real and fake.

    Args:
        pred_real (Tensor): Discriminator output for real samples.
        pred_fake (Tensor): Discriminator output for fake samples.

    Returns:
        Tensor: Combined BCE loss.
    """
    loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
    loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    return 0.5 * (loss_real + loss_fake)
