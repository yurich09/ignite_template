from loguru import logger


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
