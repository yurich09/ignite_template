import torch
import torch.nn.functional as F
from torch.nn import Module

from .base import confusion_mat_grad, dice


def _dice_loss(pred, true):
    mat = confusion_mat_grad(pred, true)
    mat = mat.double() / mat.sum()
    return 1 - dice(mat).mean()


class DiceLoss(Module):
    def forward(self, inputs, targets):
        return _dice_loss(inputs, targets)


class WeightLoss(Module):
    def forward(self, inputs, targets):
        with torch.autocast('cuda'):  # Always use fp32
            cce = F.cross_entropy(inputs, targets)

        dice_loss = _dice_loss(inputs, targets)
        return cce + 0.5 * dice_loss
