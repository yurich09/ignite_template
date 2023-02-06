import torch
import torch.nn.functional as F
from torch.nn import Module

from .base import dice, soft_confusion, to_probs


def _dice_loss(pred, true):
    c, pred, true = to_probs(pred, true)
    mat = soft_confusion(c, pred, true)

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
