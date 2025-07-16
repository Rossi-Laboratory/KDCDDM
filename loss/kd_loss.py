import torch.nn.functional as F

def kd_l1_loss(student_output, teacher_output):
    """
    L1 loss for knowledge distillation.

    Args:
        student_output (Tensor): Output from student model.
        teacher_output (Tensor): Target output from teacher model.

    Returns:
        Tensor: L1 loss.
    """
    return F.l1_loss(student_output, teacher_output.detach())

def kd_l2_loss(student_output, teacher_output):
    """
    L2 loss for knowledge distillation.

    Args:
        student_output (Tensor): Output from student model.
        teacher_output (Tensor): Target output from teacher model.

    Returns:
        Tensor: L2 (MSE) loss.
    """
    return F.mse_loss(student_output, teacher_output.detach())
