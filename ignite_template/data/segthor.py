from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import ignite.distributed as idist
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def _split(folder: str, split_coef: float) -> Tuple[List, ...]:
    gt = Path(folder).rglob('GT.nii.gz')
    data = [(f'{p.parent}\\{p.parent.name}.nii.gz', str(p)) for p in gt]
    k_split = int(len(data) * split_coef)
    return data[k_split:], data[:k_split]


def sub_idiv(x: np.ndarray, sub: np.ndarray, div: np.ndarray) -> np.ndarray:
    """Efficiently do `(x - sub) / div`. Output is `f4`"""
    #  sub -> buf[f32] -> idiv
    x = np.subtract(x, sub, dtype='f4')
    x /= div
    return x


def norm_zero_to_one(
    a: np.ndarray,
    a_min: Optional[float] = None,
    a_max: Optional[float] = None,
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


def resize(data: np.ndarray,
           order: int,
           shape: List[int],
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


def _load(path: str, clip: List[int], shape: List[int]) -> np.ndarray:
    arr = np.asanyarray(nib.load(path).dataobj)

    # xyz -> zxy transpose
    arr = arr.transpose(2, 0, 1)

    # clip, norm by 1 and resize to shape with cubic
    if 'GT.nii.gz' not in path:
        arr = arr.clip(*clip)
        arr = norm_zero_to_one(arr, clip[0], clip[1])
        arr = resize(arr, 3, shape)
    # resize to shape with nearest
    else:
        arr = resize(arr, 0, shape)

    return arr


@dataclass(frozen=True)
class SegThor(Dataset):
    data: List
    clip: List[int]
    shape: List[int]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img, mask = [
            _load(item, self.clip, self.shape) for item in self.data[idx]
        ]

        # c z x y
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), torch.from_numpy(mask).long()


def get_loaders(folder: str, split_coef: float, batch: int, workers: int,
                shape: List[int], clip: List[int]):
    train_set, valid_set = _split(folder, split_coef)
    return [
        idist.auto_dataloader(
            dataset=SegThor(item, clip, shape),
            batch_size=batch,
            num_workers=workers,
        ) for item in (train_set, valid_set)
    ]
