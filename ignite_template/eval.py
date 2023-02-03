import json
from pathlib import Path
from typing import NamedTuple

import hydra
import omegaconf
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module

DEVICE = torch.device('cuda:0')

class OutParams(NamedTuple):
    cfg: str
    split: str
    ckpt: str

def main():
    pass

class Eval:
    def __init__(self, params: OutParams):

        with open(params.split) as f:
            split = json.load(f)
            self.tsplit = split['train']
            self.vsplit = split['val']

        self.cfg = omegaconf.OmegaConf.load(params.cfg)

        logger.info(f'Creating <{self.cfg.model._target_}>')
        self.net: Module = hydra.utils.instantiate(self.cfg.model)

        logger.info(f'Load model weights')
        self.net.load_state_dict(torch.load(params.ckpt))
        self.net.to(DEVICE)
        self.net.eval()

        logger.info(f'Creating <{self.cfg.data.eval._target_}>')
        self.loader = hydra.utils.instantiate(self.cfg.data.eval, data=self.vsplit)

        self.metrcs = []

    def predict_all(self, ):
        pass

    def _predict(self, arr: torch.Tensor):
        with torch.no_grad():
            pred = self.net()


def make_outprams(folder = './outputs/'):
    out = []
    outputs = Path(folder).iterdir()
    outputs = [p for sublist in outputs for p in sublist.iterdir()]
    outputs = [p for p in outputs if not Path(f'{p}/metrics.json').exists()]
    for path in outputs:
        try:
            ckpt = next(path.glob('*.pt'))
            out.append(OutParams(
                f'{path}/.hydra/config.yaml',
                f'{path}/split.json',
                str(ckpt)))
        except StopIteration:
            continue

    return out


if __name__ == '__main__':
    for params in make_outprams():
        evaluator = Eval(params)
