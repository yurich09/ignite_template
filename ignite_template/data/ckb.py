from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import albumentations as A
import cv2
import ignite.distributed as idist
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as F


class CkbDataset(Dataset):
    def __init__(self, aug: bool) -> None:
        self.aug = aug

    def __len__(self, ) -> int:
        return 1

    def __getitem__(self, index: tuple) -> tuple[torch.Tensor, ...]:
        img, mask = index
        bgr = cv2.imread(img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask)[:, :, 0]
        if self.aug:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.Blur(p=0.15),
                A.MedianBlur(p=0.15),
                A.RandomGamma(p=0.15),
                A.Transpose(p=0.5),
            ])
            transformed = transform(image=rgb, mask=mask)
            rgb = transformed['image']
            mask = transformed['mask']
        return F.to_tensor(rgb), torch.from_numpy(mask).long()


class RandomSampler(Sampler):
    def __init__(self, data: Sequence[tuple[Path, Path]], samples: int):
        self.data = data
        self.samples = samples
        self._rg = random.Random(None)

    def __len__(self,) -> int:
        return self.samples

    def __iter__(self,) -> Iterator[tuple]:
        for _ in range(len(self)):
            img, mask = self._rg.choice(self.data)
            yield str(img), str(mask)


def get_datasets(aug: bool) -> tuple[CkbDataset, ...]:
    assert idist.get_rank() == 0

    return CkbDataset(aug), CkbDataset(aug), CkbDataset(False)


def get_loaders(subsets: tuple[Dataset, Dataset, Dataset], batch: int,
                workers: int, samples: int, train: str, val: str):
    tset, tvset, vset = subsets

    t = [(p, p.with_suffix('.mask.png')) for p in Path(train).glob('*cancer.jpeg') if p.exists() and p.with_suffix('.mask.png').exists()]
    v = [(p, p.with_suffix('.mask.png')) for p in Path(val).glob('*cancer.jpeg') if p.exists() and p.with_suffix('.mask.png').exists()]
    print(len(t), len(v))
    return [
        idist.auto_dataloader(
            dset,
            sampler=RandomSampler(data, samples),
            batch_size=batch,
            num_workers=workers,
        ) for dset, data in [(tset, t), (tvset, v), (vset, v)]
    ]
