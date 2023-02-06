from __future__ import annotations

from typing import Any

import hydra
import torch
from ignite.metrics import Loss, Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from loguru import logger
from omegaconf import DictConfig
from torch.nn import Module

from .base import confusion, dice, to_indices


class Dice(Metric):
    def __init__(self,
                 num_classes: int,
                 output_transform=lambda x: x,
                 device='cpu'):
        self._mat: torch.Tensor | None = None
        self._num_classes = num_classes
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        c = self._num_classes
        self._mat = torch.zeros(c, c, device=self._device, dtype=torch.long)

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        pred, true = (o.detach() for o in output)
        c, pred, true = to_indices(pred, true)
        self._mat += confusion(c, pred, true).to(self._device)

    @sync_all_reduce('_mat:SUM')
    def compute(self) -> float | torch.Tensor:
        assert self._mat is not None
        mat = self._mat.double() / self._mat.sum()
        return dice(mat).mean()


def make_metrics(cfg: DictConfig, loss: Module) -> dict[str, Any]:
    logger.info(f'Add loss <{loss.__module__}>')
    metrics = {'loss': Loss(loss)}
    if not cfg:
        logger.warning('No callback configs found! Skipping..')
        return metrics

    if not isinstance(cfg, DictConfig):
        raise TypeError('Callbacks config must be a DictConfig!')

    for key, cb_conf in cfg.items():
        if isinstance(cb_conf, DictConfig) and '_target_' in cb_conf:
            logger.info(f'Add metric <{cb_conf._target_}>')
            metrics[str(key)] = hydra.utils.instantiate(cb_conf)

    return metrics
