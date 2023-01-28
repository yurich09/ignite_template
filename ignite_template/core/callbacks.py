from collections import defaultdict
from pathlib import Path
from typing import Any

from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from loguru import logger
from torch.nn import Module


class Eval:
    def __init__(self, tloader, vloader, evaluator, score_name: str):
        self._loaders = {'train': tloader, 'valid': vloader}
        self._evaluator = evaluator
        self._sota = float('inf' if score_name.endswith('loss') else '-inf')
        self._score_name = score_name

    def __call__(self, engine):
        epoch_metrics: dict[str, list[str]] = defaultdict(list)

        for i, (mode, loader) in enumerate(self._loaders.items()):
            metrics: dict[str, float] = self._evaluator.run(loader).metrics
            for name, value in metrics.items():
                engine.state.metrics[f'{mode}_{name}'] = value
                epoch_metrics[name].append(f'{value:.4f}')

        is_sota = (value < self._sota) if self._score_name.endswith('loss') else (value > self._sota)
        if is_sota:
            self._sota = value

        metrics_str = ', '.join(
            f'{name}: {{}}'.format('/'.join(values))
            for name, values in epoch_metrics.items()
        )
        emit = logger.info if is_sota else logger.debug
        emit(f'[{engine.state.epoch:03d}] {metrics_str}')


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
