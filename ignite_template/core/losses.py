import torch.nn.functional as F
from torch.nn import Module

from ignite_template.core.base import dice


class DiceLoss(Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        return (1 - dice(inputs, targets)).mean()


class WeightLoss(Module):

    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, inputs, targets):
        cce = F.cross_entropy(inputs, targets)
        dice_loss = 1 - dice(inputs, targets)
        return (cce + 0.5 * dice_loss).mean()
