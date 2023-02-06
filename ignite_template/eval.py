import copy
import json
from collections.abc import Iterator, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from torch.nn import Module
from tqdm.auto import tqdm

from ignite_template.core.metrics import confusion, dice
from ignite_template.data.segthor import get_val_loader

DEVICE = torch.device('cuda:0')
NUM_GPUS = torch.cuda.device_count()
FP16 = any(
    torch.cuda.get_device_capability(dev) > (7, 0) for dev in range(NUM_GPUS))


class RunParams(NamedTuple):
    root: Path
    cfg: Path
    split: Path
    ckpt: Path


def iter_runs(folder: str) -> Iterator[RunParams]:
    for fdr in Path(folder).rglob('.hydra'):
        root = fdr.parent
        cfg_path = fdr / 'config.yaml'
        split_path = root / 'split.json'
        if not cfg_path.is_file() or not split_path.is_file():
            continue
        if not (ckpt := next(root.glob('*.pt'), None)):
            continue
        yield RunParams(root, cfg_path, split_path, ckpt)


def _nd_resize_sample(obj: torch.Tensor,
                      shape_src: tuple[int, ...]) -> torch.Tensor:
    """Resize `obj` to have shape of `shape_src`. Batch should not present"""
    rank, mode = {
        2: (1, 'linear'),  # (c n)
        3: (2, 'bilinear'),  # (c h w)
        4: (3, 'trilinear'),  # (c d h w)
    }[obj.ndim]
    size = shape_src[-rank:]
    return F.interpolate(obj[None], size, mode=mode, align_corners=True)[0]


class _Forward(NamedTuple):
    net: Module
    device: int
    lock: Lock

    @torch.autocast('cuda', enabled=FP16)
    @torch.inference_mode()
    def compute_cm(self, sample_ids: Sequence[str], xs: torch.Tensor,
                   ys: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        s = torch.cuda.current_stream(self.device)
        with self.lock, torch.cuda.device(self.device), torch.cuda.stream(s):
            # Predict
            ys = [y.cuda(non_blocking=True) for y in ys]
            y_preds = self.net(xs.cuda(non_blocking=True))

            mats = []
            for y, y_pred in zip(ys, y_preds):
                c = y_pred.shape[0]

                # Resize to original size
                y_pred = _nd_resize_sample(y_pred, y.shape).argmax(0)

                # Compute matrix
                mats.append(confusion(c, y_pred, y))

            mats_ = torch.stack(mats).cpu()
        return dict(zip(sample_ids, mats_.unbind()))


class SegTester:
    def __init__(self, params: RunParams):
        basecfg = OmegaConf.load(params.cfg)

        logger.info(f'Creating <{basecfg.model._target_}>')
        net: Module = hydra.utils.instantiate(basecfg.model)

        logger.info('Load model weights')
        net.load_state_dict(torch.load(params.ckpt))
        net.eval()

        self.fwds = [
            _Forward(copy.deepcopy(net).to(dev), dev, Lock())
            for dev in range(NUM_GPUS)
        ]

        logger.info('Creating dataloader')
        self.loader = get_val_loader(
            shape=basecfg.data.prepare.shape,
            hu_range=basecfg.data.prepare.hu_range,
            data=json.loads(params.split.read_text())['val'],
        )

    def compute_scores(self, ex: Executor) -> tuple[pd.DataFrame, dict]:
        # Some multi-GPU trickery
        fs = [
            ex.submit(self.fwds[i % NUM_GPUS].compute_cm, paths, vois, masks)
            for i, (paths, vois,
                    masks) in enumerate(tqdm(self.loader, leave=False))
        ]
        wait(fs, return_when='ALL_COMPLETED')
        mats = {path: mat for f in fs for path, mat in f.result().items()}

        # Compute relevant metrics
        rows = {
            path: {f'dice_{i}': v for i, v in enumerate(dice(mat).tolist())}
            for path, mat in mats.items()
        }
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = 'path'

        dice_cols = [name for name in df.columns if name.startswith('dice')]
        df['dice'] = df[dice_cols].sum(axis=1)

        # Compute summaries
        summary = {}
        for name in df.columns:
            if name == 'path':  # Skip index
                continue
            col = df[name].dropna()
            summary[f'min/{name}'] = col.min()
            summary[f'mean/{name}'] = col.mean()
            summary[f'max/{name}'] = col.min()

        return df, summary


def main():
    # Setup loguru
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)

    # Find runs to evaluate
    runs = [rp for fdr in ('outputs', 'multirun') for rp in iter_runs(fdr)]
    with ThreadPoolExecutor(NUM_GPUS * 2) as ex:
        for run in tqdm(runs):
            outfile = run.root / 'metrics.csv'
            if outfile.is_file():  # Skip done runs
                continue

            logger.info(f'Predict run {run.root}')
            df, obj = SegTester(run).compute_scores(ex)

            df.to_csv(outfile, float_format='%.4f')
            with (run.root / 'summary.json').open('w') as fp:
                json.dump(obj, fp, indent=2)


if __name__ == '__main__':
    main()
