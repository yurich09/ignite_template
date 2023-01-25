from pathlib import Path
from typing import Any

from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from loguru import logger
from torch.nn import Module


class Eval:

    def __init__(self, tloader, vloader, evaluator):
        self.loaders = {'train': tloader, 'valid': vloader}
        self.evaluator = evaluator

    def __call__(self, engine):
        for mode, loader in self.loaders.items():
            metrics: dict[str, float] = self.evaluator.run(loader).metrics
            metrics_str: dict[str, str] = {}
            for key, value in metrics.items():
                engine.state.metrics[f'{mode}_{key}'] = value
                metrics_str[f'{mode}_{key}'] = f'{value:.2f}'
            logger.info(
                f'Epoch: {engine.state.epoch}. metrics: {str(metrics_str)}')


class Saver:

    def __init__(self, net: Module, trainer: Any):
        self.saver = Checkpoint(
            {'model': net},
            DiskSaver(Path()),
            n_saved=1,
            filename_prefix='best',
            score_function=lambda engine: engine.state.metrics['valid_dice'],
            score_name='valid_dice',
            global_step_transform=global_step_from_engine(trainer))

    def __call__(self, engine):
        return self.saver(engine)
