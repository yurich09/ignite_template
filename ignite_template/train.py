import hydra
import ignite.distributed as idist
from ignite.contrib.handlers import tqdm_logger
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import RunningAverage
from ignite.utils import manual_seed
from loguru import logger
from omegaconf import DictConfig
from torch.nn import Module

from ignite_template.core.callbecks import Ckpt, Eval
from ignite_template.core.metrics import make

logger.disable("ignite")


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="train.yaml")
def main(cfg: DictConfig):

    if cfg.seed:
        manual_seed(cfg.seed)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    tloader, vloader = hydra.utils.instantiate(cfg.data)

    logger.info(f'Instantiating model <{cfg.model._target_}>')
    net: Module = hydra.utils.instantiate(cfg.model)
    net = idist.auto_model(net, sync_bn=True)

    logger.info(f'Instantiating loss functions <{cfg.loss._target_}>')
    loss: Module = hydra.utils.instantiate(cfg.loss)

    logger.info(f'Instantiating optimizer <{cfg.optim._target_}>')
    optimizer = hydra.utils.instantiate(cfg.optim, params=net.parameters())

    metrics = make(cfg.metrics, loss=loss)

    trainer = create_supervised_trainer(net, optimizer, loss, cfg.device)
    evaluator = create_supervised_evaluator(net, metrics, cfg.device)

    validator = Eval(tloader, vloader, evaluator)
    saver = Ckpt(trainer)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    if idist.get_rank() == 0:
        tqdm_logger.ProgressBar().attach(trainer, ['loss'])
        tqdm_logger.ProgressBar().attach(evaluator)

    @trainer.on(Events.ITERATION_COMPLETED)
    def _finilize_iterations(engine):
        pass

    @trainer.on(Events.EPOCH_COMPLETED)
    def _finilize_epoch(engine):
        validator(engine)
        saver(engine, net)

    trainer.run(tloader, max_epochs=cfg.epoch)


if __name__ == '__main__':
    main()
