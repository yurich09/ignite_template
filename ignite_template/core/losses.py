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
        return (F.cross_entropy(inputs, targets)
                + 0.5 * _dice_loss(inputs, targets))
