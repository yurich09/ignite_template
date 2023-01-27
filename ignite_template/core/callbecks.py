from collections import defaultdict
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
        epoch_metrics: dict[str, list[str]] = defaultdict(list)

        for mode, loader in self.loaders.items():
            metrics: dict[str, float] = self.evaluator.run(loader).metrics
            for key, value in metrics.items():
                engine.state.metrics[f'{mode}_{key}'] = value
                epoch_metrics[key].append(f'{value:.4f}')

        metrics_str = ', '.join(
            f'{key}: {{}}'.format('/'.join(values))
            for key, values in epoch_metrics.items()
        )
        logger.info(f'[{engine.state.epoch:03d}] {metrics_str}')


class Saver:
    def __init__(self, net: Module, trainer: Any, score_name: str):
        self.saver = Checkpoint(
            {'model': net},
            DiskSaver(Path()),
            n_saved=1,
            filename_prefix='best',
            score_function=lambda engine: engine.state.metrics[score_name],
            score_name=score_name,
            global_step_transform=global_step_from_engine(trainer),
        )

    def __call__(self, engine):
        return self.saver(engine)
