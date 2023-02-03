from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


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


def load_zyx(path: Path) -> tuple[np.ndarray, Any]:
    obj = nib.load(str(path))
    xyz = np.asanyarray(obj.dataobj)
    return xyz.T, obj.affine


def save_zyx(path: Path, zyx: np.ndarray, mat) -> None:
    if zyx.dtype in ['i8', 'i4']:
        zyx = zyx.astype('i2')
    elif zyx.dtype == 'f8':
        zyx = zyx.astype('f4')

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(zyx.T, mat).to_filename(str(path))
