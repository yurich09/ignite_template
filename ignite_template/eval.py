import json
from pathlib import Path
from typing import NamedTuple

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module

from ignite_template.core.metrics import confusion_mat, dice

DEVICE = torch.device('cuda:0')

class OutParams(NamedTuple):
    root: Path
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

    def compute_dice(self) -> pd.DataFrame:
        rows = []
        for path, arr, mask in self.loader:
            logger.info(f'Run predict on {path}')
            pred = self._predict(arr, mask.shape[-3:]).cpu()
            cm = confusion_mat(pred, mask)
            dice_value = dice(cm).numpy().tolist()
            rows.append({'path': path[0], 'dice_mean': f'{np.mean(dice_value):.4f}'} | {f'dice{i}': f'{v:.4f}' for i, v in enumerate(dice_value)})

        return pd.DataFrame.from_records(rows).set_index('path')




    def _predict(self, x: torch.Tensor, shape: tuple) -> np.ndarray:
        with torch.no_grad():
            pred = self.net(x.to(DEVICE))
        pred = F.interpolate(pred, shape, mode='trilinear', align_corners=False)
        return pred


def make_outprams(folder = './outputs/'):
    out = []
    outputs = Path(folder).iterdir()
    outputs = [p for sublist in outputs for p in sublist.iterdir()]
    outputs = [p for p in outputs if not (p / 'metrics.csv').exists()]
    for path in outputs:
        try:
            ckpt = next(path.glob('*.pt'))
            out.append(OutParams(
                path,
                f'{path}/.hydra/config.yaml',
                f'{path}/split.json',
                str(ckpt)))
        except StopIteration:
            continue

    return out


if __name__ == '__main__':
    for params in make_outprams():
        evaluator = Eval(params)
        df = evaluator.compute_dice()
        df.to_csv(params.root / 'metrics.csv')
