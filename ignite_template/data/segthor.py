from __future__ import annotations

import json
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ignite.distributed as idist
import numpy as np
import torch
from loguru import logger
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from ignite_template.core.data import clip_and_norm, load_zyx, resize, save_zyx


def _find_root(dataroot: Path) -> Path:
    first = next(dataroot.rglob('GT.nii.gz'))
    return first.parent.parent


def _split(dataroot: Path, val_ratio: float,
           seed: int | None) -> tuple[list[Path], ...]:
    assert idist.get_rank() == 0

    folders = sorted(dataroot.iterdir())
    if seed is None:
        logger.warning('Split seed not set. Train/val split is randomized')
    random.Random(seed).shuffle(folders)

    pos = int(len(folders) * val_ratio)
    subsets = folders[pos:], folders[:pos]

    obj = {
        k: [p.as_posix() for p in ps]
        for k, ps in zip(('train', 'val'), subsets)
    }
    with Path('split.json').open('w') as fp:
        json.dump(obj, fp, indent=2)

    return subsets


warnings.filterwarnings('ignore', module='scipy.ndimage')


@dataclass(frozen=True)
class ShiftScale:
    maxscale: float = 1.1
    maxshift: float = 0.05

    def __bool__(self) -> bool:
        return self.maxscale > 1 or self.maxshift > 0

    def __call__(self, *arrs: np.ndarray) -> tuple[np.ndarray, ...]:
        assert len({a.shape for a in arrs}) == 1
        if not self:
            return arrs

        rank = arrs[0].ndim
        kx, dx = (torch.rand(2, rank).numpy() * 2 - 1)
        scale = (self.maxscale ** kx) if self.maxscale > 1 else np.ones(3)
        shift = ((self.maxshift * dx * arrs[0].shape[:rank])
                 if self.maxshift > 0 else np.zeros(3))

        center = np.array(arrs[0].shape) / 2
        offset = center * (1 - scale) + shift

        rs: tuple[np.ndarray, ...] = ()
        for a in arrs:
            if a.dtype.kind == 'f':
                order = 3  # cubic
                a = ndimage.spline_filter(a, order=order, output='f4')
            else:
                order = 0  # nearest
            r = ndimage.affine_transform(
                a, scale, offset, order=order, prefilter=False).astype(a.dtype)
            rs += r,
        return rs


def _load_voi(path: Path, hu_range: tuple[int, int],
              shape: Sequence[int]) -> tuple[np.ndarray, Any]:
    arr, mat = load_zyx(path)
    arr = clip_and_norm(arr, *hu_range)
    arr = resize(arr, 3, shape)  # cubic resize
    return arr, mat


def _load_gt(path: Path, shape: Sequence[int]) -> tuple[np.ndarray, Any]:
    arr, mat = load_zyx(path)
    arr = resize(arr, 0, shape)  # nearest resize
    return arr, mat


@dataclass(frozen=True)
class NiftyCacher(Dataset):
    caches: tuple[Path, Path]
    folders: Sequence[Path]
    hu_range: tuple[int, int]
    shape: Sequence[int]

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx: int) -> tuple[Path, Path]:
        fdr = self.folders[idx]
        from_voi = fdr / f'{fdr.name}.nii.gz'
        from_mask = fdr / 'GT.nii.gz'

        to_voi, to_mask = (
            cache / f'{fdr.name}.nii.gz' for cache in self.caches)

        # Save volume
        if not to_voi.is_file():
            voi, mat = _load_voi(from_voi, self.hu_range, self.shape)
            save_zyx(to_voi, voi, mat)

        # Save mask
        if not to_mask.is_file():
            mask, mat = _load_gt(from_mask, self.shape)
            save_zyx(to_mask, mask, mat)

        return (to_voi, to_mask)


class NiftyDataset(Dataset):
    def __init__(self,
                 files: Sequence[tuple[Path, Path]],
                 num_reps: int = 1,
                 aug: Any | None = None) -> None:
        self.files = files
        self.num_reps = num_reps
        self.aug = aug

    def __len__(self) -> int:
        return len(self.files) * self.num_reps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        voi_path, mask_path = self.files[idx % len(self.files)]
        voi, _ = load_zyx(voi_path)
        mask, _ = load_zyx(mask_path)

        if self.aug:
            voi, mask = self.aug(voi, mask)

        voi = voi[None, ...]  # c z x y
        return torch.from_numpy(voi), torch.from_numpy(mask).long()


def get_datasets(
    folder: str,
    val_ratio: float,
    shape: Sequence[int],
    hu_range: tuple[int, int],
    seed: int | None = None,
    num_reps: int = 1,
    maxscale: float = 1,
    maxshift: float = 0,
    valaug: bool = False,
) -> tuple[NiftyDataset, NiftyDataset, NiftyDataset]:
    assert idist.get_rank() == 0

    dataroot = _find_root(Path(folder))
    train_dirs, valid_dirs = _split(dataroot, val_ratio, seed)

    cacheroot = dataroot.parent / f'cache_{shape[0]}_{shape[1]}_{shape[2]}'
    caches = (cacheroot / f'hu_{hu_range[0]}_{hu_range[1]}',
              cacheroot / 'masks')

    # Generate small volumes to optimize disk reads
    fsets = (
        NiftyCacher(caches, dirs, hu_range, shape)
        for dirs in (train_dirs, valid_dirs))

    loaders = (DataLoader(fset, None, num_workers=8) for fset in fsets)

    train_files, valid_files = (
        list(tqdm(loader, desc='Generate volume rescales'))
        for loader in loaders)

    # Create final datasets
    taug = ShiftScale(maxscale=maxscale, maxshift=maxshift)
    tset = NiftyDataset(train_files, num_reps=num_reps, aug=taug)

    vaug, vreps = (taug, 4) if taug and valaug else (None, 1)
    tvset = NiftyDataset(train_files, num_reps=vreps, aug=vaug)
    vset = NiftyDataset(valid_files, num_reps=vreps, aug=vaug)

    return tset, tvset, vset


def get_loaders(subsets: tuple[Dataset, Dataset, Dataset], batch: int,
                workers: int):
    tset, tvset, vset = subsets
    return [
        idist.auto_dataloader(
            dset,
            batch_size=batch,
            num_workers=workers,
            shuffle=is_train,
        ) for is_train, dset in [(True, tset), (False, tvset), (False, vset)]
    ]


class EvalDataset(Dataset):
    def __init__(self, data: Sequence[str], shape: tuple[int, ...],
                 hu_range: tuple[int, int]) -> None:
        self.shape = shape
        self.hu_range = hu_range
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        folder = Path(self.data[idx])
        voi_path = folder / f'{folder.name}.nii.gz'
        mask_path = folder / 'GT.nii.gz'
        voi, _ = load_zyx(voi_path)
        mask, _ = load_zyx(mask_path)

        voi = clip_and_norm(voi, *self.hu_range)
        voi = resize(voi, 3, self.shape)

        voi = voi[None, ...]  # c z y x
        return (str(folder), torch.from_numpy(voi),
                torch.from_numpy(mask).long())


def _collate_fn_internal(batch):
    if (all(isinstance(x, (torch.Tensor, np.ndarray)) for x in batch)
            and len({x.shape for x in batch}) > 1):
        return [torch.as_tensor(x) for x in batch]
    return default_collate(batch)


def _collate_fn(batch):
    groups = *zip(*batch),
    return *(_collate_fn_internal(group) for group in groups),


def get_val_loader(shape: tuple[int, ...], hu_range: tuple[int, int],
                   data: Sequence[str]):
    return DataLoader(
        EvalDataset(data, shape, hu_range),
        batch_size=4,
        num_workers=8,
        collate_fn=_collate_fn,
    )
