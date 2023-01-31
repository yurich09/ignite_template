import os

import hydra
import ignite.distributed as idist
import torch
from ignite.contrib.handlers import tqdm_logger
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import RunningAverage
from ignite.utils import manual_seed
from loguru import logger
from omegaconf import DictConfig
from torch.nn import Module

from ignite_template.core.callbacks import Eval, get_saver
from ignite_template.core.metrics import make_metrics

NUM_GPUS = torch.cuda.device_count()
DIST_PARAMS = {'backend': 'nccl', 'nproc_per_node': NUM_GPUS}


def train(rank, cfg: DictConfig, tset, vset, device):
    if cfg.seed:
        manual_seed(cfg.seed)
    if rank == 0:
        logger.add('log.log', level='DEBUG')
    else:
        logger.info(f'Mute logger in {os.getpid()} process')
        logger.remove()

    logger.info(f'Creating <{cfg.data.loaders._target_}>')
    tloader, t2loader, vloader = hydra.utils.instantiate(
        cfg.data.loaders,
        tset=tset,
        vset=vset,
    )

    logger.info(f'Creating <{cfg.model._target_}>')
    net: Module = hydra.utils.instantiate(cfg.model)

    num_params = sum(
        p.numel() for it in (net.parameters(), net.buffers()) for p in it)
    logger.info(f'Total parameters: {num_params/1e6:.2f}M')

    if device == 'cuda':
        net = idist.auto_model(net, sync_bn=True)
    else:
        net.to(device)

    logger.info(f'Creating <{cfg.loss._target_}>')
    loss: Module = hydra.utils.instantiate(cfg.loss)

    logger.info(f'Creating <{cfg.optim._target_}>')
    optimizer = hydra.utils.instantiate(cfg.optim, params=net.parameters())

    logger.info(f'Creating <{cfg.sched._target_}>')
    scheduler = hydra.utils.instantiate(
        cfg.sched,
        optimizer=optimizer,
        epochs=cfg.epoch,
        steps_per_epoch=len(tloader),
    )

    metrics = make_metrics(cfg.metrics, loss=loss)

    load_kwargs = {'device': device, 'non_blocking': True}
    if cfg.fp16:
        logger.info('FP16 is enabled')
        amp, scaler = 'amp', True
    else:
        amp, scaler = None, False

    trainer = create_supervised_trainer(
        net, optimizer, loss, **load_kwargs, amp_mode=amp, scaler=scaler)
    evaluator = create_supervised_evaluator(
        net, metrics, **load_kwargs, amp_mode=amp)

    score_name = 'valid_dice'
    validate = Eval(t2loader, vloader, evaluator, score_name=score_name)
    save = get_saver(net, trainer, score_name=score_name)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    if rank == 0:
        tqdm_logger.ProgressBar().attach(trainer, ['loss'])
        tqdm_logger.ProgressBar().attach(evaluator)

    @trainer.on(Events.ITERATION_COMPLETED)
    def _finalize_iterations(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def _finalize_epoch(engine):
        validate(engine)
        if rank == 0:
            save(engine)

    trainer.run(tloader, max_epochs=cfg.epoch)


@hydra.main('../configs', 'train.yaml', '1.3')
def main(cfg):
    tset, vset = hydra.utils.instantiate(cfg.data.prepare)

    if cfg.device == 'cuda' and NUM_GPUS > 1:
        with idist.Parallel(**DIST_PARAMS) as parallel:
            parallel.run(train, cfg, tset, vset, cfg.device)
    else:
        train(0, cfg, tset, vset, cfg.device)


if __name__ == '__main__':
    main()
