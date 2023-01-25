from typing import Any, Dict, Sequence, Union

import hydra
import torch
from ignite.metrics import Loss, Metric
from loguru import logger
from omegaconf import DictConfig
from torch.nn import Module

from ignite_template.core.base import dice


class Dice(Metric):

    def reset(self) -> None:
        self.count = 0
        self.sum = torch.tensor(0.0, device=self._device)

    def update(self, output: Sequence[torch.Tensor]) -> None:
        pred, true = output[0].detach(), output[1].detach()
        self.sum += dice(pred, true).mean().to(self._device)
        self.count += 1

    def compute(self) -> Union[float, torch.Tensor]:
        return self.sum.item() / self.count


def make_metrics(cfg: DictConfig, loss: Module) -> Dict[str, Any]:
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
