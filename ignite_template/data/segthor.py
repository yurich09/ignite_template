from __future__ import annotations

import json
import os
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ignite.distributed as idist
import nibabel as nib
import numpy as np
import torch
from loguru import logger
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _find_root(dataroot: Path) -> Path:
    first = next(dataroot.rglob('GT.nii.gz'))
    return first.parent.parent


def _split(dataroot: Path, ratio: float,
           seed: int | None) -> tuple[list[Path], ...]:
    folders = sorted(dataroot.iterdir())

    random.Random(seed).shuffle(folders)

    pos = int(len(folders) * ratio)
    subsets = folders[pos:], folders[:pos]

    obj = {
        k: [p.as_posix() for p in ps]
        for k, ps in zip(('train', 'val'), subsets)
    }
    with Path(f'split.{os.getpid()}.json').open('w') as fp:
        json.dump(obj, fp, indent=2)

    return subsets


def sub_idiv(x: np.ndarray, sub: np.ndarray, div: np.ndarray) -> np.ndarray:
    """Efficiently do `(x - sub) / div`. Output is `f4`"""
    #  sub -> buf[f32] -> idiv
    x = np.subtract(x, sub, dtype='f4')
    x /= div
    return x


def norm_zero_to_one(
    a: np.ndarray,
    a_min: float | None = None,
    a_max: float | None = None,
    axes=None,
) -> np.ndarray:
    """
    Clips data to [min ... max] range, then remaps to [0 ... 1] range
    """
    if a_min is None:
        a_min = a.min(axes, keepdims=True)
    if a_max is None:
        a_max = a.max(axes, keepdims=True)
    return sub_idiv(a, a_min, a_max - a_min)  # type: ignore


def clip_and_norm(arr: np.ndarray,
                  min_clip: float,
                  max_clip: float,
                  axes=None) -> np.ndarray:
    arr = arr.clip(min_clip, max_clip)
    return norm_zero_to_one(arr, min_clip, max_clip, axes=axes)


def _resize(data: np.ndarray,
            zoom: Sequence[float],
            order: int,
            antialias: bool = False) -> np.ndarray:
    """
    This function resizes input data with specified interpolation mode.

    Parameters:
    - data - data to process it
    - order - The interpolation order
      (0 - nearest, 1 - linear, 2 - quadratic, 3 - cubic, etc.., up to 5)
    - antialias - set to use gaussian smoothing to reduce high frequencies and
      suppress aliasing.
    """
    if antialias:
        data = ndimage.gaussian_filter(data, sigma=0.35 / np.array(zoom))

    return ndimage.zoom(data, zoom=zoom, order=order, prefilter=False)


def resize(data: np.ndarray,
           order: int,
           shape: Sequence[int],
           antialias: bool = False) -> np.ndarray:
    """
    This function resizes input data with specified interpolation mode.

    Parameters:
    - data - data to process it
    - order - The interpolation order
      (0 - nearest, 1 - linear, 2 - quadratic, 3 - cubic, etc.., up to 5)
    - shape - The target shape
    - antialias - set to use gaussian smoothing to reduce high frequencies and
      suppress aliasing.
    """
    zoom = [dst / src for dst, src in zip(shape, data.shape)]
    return _resize(data, zoom, order, antialias)


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


def _load_zxy(path: Path) -> tuple[np.ndarray, Any]:
    obj = nib.load(str(path))
    xyz = np.asanyarray(obj.dataobj)
    return xyz.transpose(2, 0, 1), obj.affine


def _load_voi(path: Path, hu_range: tuple[int, int],
              shape: Sequence[int]) -> tuple[np.ndarray, Any]:
    arr, mat = _load_zxy(path)
    arr = clip_and_norm(arr, *hu_range)
    arr = resize(arr, 3, shape)  # cubic resize
    return arr, mat


def _load_gt(path: Path, shape: Sequence[int]) -> tuple[np.ndarray, Any]:
    arr, mat = _load_zxy(path)
    arr = resize(arr, 0, shape)  # nearest resize
    return arr, mat


def _save_zxy(path: Path, zxy: np.ndarray, mat) -> None:
    xyz = zxy.transpose(1, 2, 0)

    if xyz.dtype in ['i8', 'i4']:
        xyz = xyz.astype('i2')
    elif xyz.dtype == 'f8':
        xyz = xyz.astype('f4')

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(xyz, mat).to_filename(str(path))


@dataclass(frozen=True)
class SegThor(Dataset):
    caches: tuple[Path, Path]
    folders: Sequence[Path]
    clip: tuple[int, int]
    shape: Sequence[int]
    num_reps: int = 1

    aug: Any | None = None

    def __len__(self) -> int:
        return len(self.folders) * self.num_reps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        fdr = self.folders[idx % len(self.folders)]

        voi_path = fdr / f'{fdr.name}.nii.gz'
        mask_path = fdr / 'GT.nii.gz'

        # Load volume
        v_cache, m_cache = (
            cache / f'{fdr.name}.nii.gz' for cache in self.caches)
        if v_cache.is_file():
            voi, _ = _load_zxy(v_cache)
        else:
            voi, mat = _load_voi(voi_path, self.clip, self.shape)
            _save_zxy(v_cache, voi, mat)

        # Load mask
        if m_cache.is_file():
            mask, _ = _load_zxy(m_cache)
        else:
            mask, mat = _load_gt(mask_path, self.shape)
            _save_zxy(m_cache, mask, mat)

        if self.aug:
            voi, mask = self.aug(voi, mask)

        voi = voi[None, ...]  # c z x y
        return torch.from_numpy(voi), torch.from_numpy(mask).long()


def get_loaders(folder: str,
                split_coef: float,
                batch: int,
                workers: int,
                shape: Sequence[int],
                clip: tuple[int, int],
                num_reps: int = 1,
                maxscale: float = 1,
                maxshift: float = 0,
                valaug: bool = False,
                seed: int | None = None,
                rank: int = 0):
    dataroot = _find_root(Path(folder))
    train_set, valid_set = _split(dataroot, split_coef, seed)

    cacheroot = dataroot.parent / f'cache_{shape[0]}_{shape[1]}_{shape[2]}'
    caches = (cacheroot / f'hu_{clip[0]}_{clip[1]}', cacheroot / 'masks')

    if rank == 0:
        dl = DataLoader(
            SegThor(caches, train_set + valid_set, clip, shape),
            batch_size=1,
            num_workers=workers,
        )
        for _ in tqdm(dl, desc='Generate volume rescales'):
            pass

    aug = ShiftScale(maxscale=maxscale, maxshift=maxshift)
    return [
        idist.auto_dataloader(
            dataset=SegThor(
                caches=caches,
                folders=folders,
                clip=clip,
                shape=shape,
                num_reps=num_reps if is_train else (4 if valaug else 1),
                aug=aug if is_train or valaug else None,
            ),
            batch_size=batch,
            num_workers=workers,
            shuffle=is_train,
        ) for is_train, folders in [
            (True, train_set),
            (False, train_set),
            (False, valid_set),
        ]
    ]
