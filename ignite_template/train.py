import hydra
import ignite.distributed as idist
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.utils import manual_seed
from loguru import logger
from omegaconf import DictConfig
from torch.nn import Module

from ignite_template.core.metrics import make


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="train.yaml")
def main(cfg: DictConfig):

    if cfg.seed:
        manual_seed(cfg.seed)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    tloader, vloader = hydra.utils.instantiate(cfg.data)
    tloader, vloader = map(idist.auto_dataloader, (tloader, vloader))

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


if __name__ == '__main__':
    main()
