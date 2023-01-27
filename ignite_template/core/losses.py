import torch.nn.functional as F
from torch.nn import Module

from ignite_template.core.base import dice_grad


class DiceLoss(Module):
    def forward(self, inputs, targets):
        return (1 - dice(inputs, targets)).mean()


class WeightLoss(Module):
    def forward(self, inputs, targets):
        cce = F.cross_entropy(inputs, targets)
        dice_loss = 1 - dice_grad(inputs, targets)
        return (cce + 0.5 * dice_loss).mean()
