from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import ignite.distributed as idist
import nibabel as nib
import numpy as np
import torch
from loguru import logger
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def _find_root(dataroot: Path) -> Path:
    first = next(dataroot.rglob('GT.nii.gz'))
    return first.parent.parent


def _split(dataroot: Path, ratio: float) -> tuple[list[Path], ...]:
    folders = sorted(dataroot.iterdir())

    pos = int(len(folders) * ratio)
    return folders[pos:], folders[:pos]


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


def clip_and_norm(arr, min_clip, max_clip, axes=None) -> np.ndarray:
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
           shape: list[int],
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


def _load_zxy(path: Path) -> tuple[np.ndarray, Any]:
    obj = nib.load(str(path))
    xyz = np.asanyarray(obj.dataobj)
    return xyz.transpose(2, 0, 1), obj.affine


def _load_voi(path: Path, hu_range: tuple[int, int], shape: tuple[int, ...]) -> tuple[np.ndarray, Any]:
    arr, mat = _load_zxy(path)
    arr = clip_and_norm(arr, *hu_range)
    arr = resize(arr, 3, shape)  # cubic resize
    return arr, mat


def _load_gt(path: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, Any]:
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
    shape: tuple[int, int, int]
    num_reps: int = 1

    def __len__(self) -> int:
        return len(self.folders) * self.num_reps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        fdr = self.folders[idx % len(self.folders)]

        voi_path = fdr / f'{fdr.name}.nii.gz'
        mask_path = fdr / 'GT.nii.gz'

        v_cache, m_cache = (cache / f'{fdr.name}.nii.gz' for cache in self.caches)
        if v_cache.is_file():
            voi, _ = _load_zxy(v_cache)
        else:
            voi, mat = _load_voi(voi_path, self.clip, self.shape)
            _save_zxy(v_cache, voi, mat)

        if m_cache.is_file():
            mask, _ = _load_zxy(m_cache)
        else:
            mask, mat = _load_gt(mask_path, self.shape)
            _save_zxy(m_cache, mask, mat)

        # c z x y
        voi = voi[None, ...]
        return torch.from_numpy(voi), torch.from_numpy(mask).long()


def get_loaders(folder: str, split_coef: float, batch: int, workers: int,
                shape: tuple[int, ...], clip: tuple[int, int], num_reps: int = 1, rank: int = 0):
    dataroot = _find_root(Path(folder))
    train_set, valid_set = _split(dataroot, split_coef)

    cacheroot = dataroot.parent / f'cache_{shape[0]}_{shape[1]}_{shape[2]}'
    caches = (cacheroot / f'hu_{clip[0]}_{clip[1]}', cacheroot / 'masks')

    if rank == 0:
        logger.info('Generate data cache')
        dl = DataLoader(
            SegThor(caches, train_set + valid_set, clip, shape),
            batch_size=1,
            num_workers=workers,
        )
        for _ in tqdm(dl):
            pass
    return [
        idist.auto_dataloader(
            dataset=SegThor(caches, folders, clip, shape, num_reps if is_train else 1),
            batch_size=batch,
            num_workers=workers,
            shuffle=is_train,
        ) for is_train, folders in [(True, train_set), (False, train_set), (False, valid_set)]
    ]
