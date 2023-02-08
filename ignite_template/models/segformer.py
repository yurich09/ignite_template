from __future__ import annotations

__all__ = [
    'SegFormer3D', 'segformer_b0', 'segformer_b1', 'segformer_b2',
    'segformer_b3', 'segformer_b4', 'segformer_b5'
]

from collections.abc import Iterable, Sequence

import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth


def _all_sum(*xs: torch.Tensor) -> torch.Tensor:
    # Single op summation
    pat = 'bcdhw'
    return torch.einsum(','.join([pat] * len(xs)) + f' -> {pat}', *xs)


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = super().forward(x)
        return rearrange(x, 'b d h w c -> b c d h w')


def mix_feedforward(dim: int, ratio: int = 4) -> nn.Sequential:
    hdim = dim * ratio
    return nn.Sequential(
        # dense
        nn.Conv3d(dim, hdim, 1, bias=False),
        # depthwise
        nn.Conv3d(hdim, hdim, 3, padding=1, groups=hdim),
        nn.GELU(),
        # dense
        nn.Conv3d(hdim, dim, 1),
    )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, tile: int = 1):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv3d(dim, dim, tile, tile),
            LayerNorm3d(dim),
        ) if tile > 1 else nn.Identity()
        self.att = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.shape
        reduced_x = self.reducer(x)

        # Attention needs tensor of shape (batch, tokens, channels)
        reduced_x = rearrange(reduced_x, 'b c d h w -> b (d h w) c')
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        out = self.att(x, reduced_x, reduced_x)[0]

        # Restore shape
        return rearrange(out, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)


class Residual(nn.Sequential):
    """Just an util layer"""
    def forward(self, x):
        return x + super().forward(x)


def emhsa_ff(dim: int,
             heads: int = 8,
             tile: int = 1,
             ff_ratio: int = 4,
             drop_path: float = .0) -> nn.Sequential:
    return nn.Sequential(
        # E-MHSA block
        Residual(
            LayerNorm3d(dim),
            EfficientMultiHeadAttention(dim, heads, tile),
            StochasticDepth(drop_path, mode='row'),
        ),
        # FeedForward block (conv-1 -> conv-3 -> conv-1)
        Residual(
            LayerNorm3d(dim),
            mix_feedforward(dim, ff_ratio),
            StochasticDepth(drop_path, mode='row'),
        ),
    )


def encoder_stage(dim_in: int,
                  dim_out: int,
                  kernel: int,
                  stride: int,
                  drop_probs: Iterable[int],
                  emha_tile: int = 1,
                  heads: int = 8,
                  ff_ratio: int = 4) -> nn.Sequential:
    if kernel % 2 == 0:  # even
        assert stride > 1
        padding = (kernel - stride) // 2
    else:  # odd
        padding = kernel // 2

    return nn.Sequential(
        # pool
        nn.Conv3d(dim_in, dim_out, kernel, stride, padding, bias=False),
        LayerNorm3d(dim_out),
        # transform
        *(emhsa_ff(dim_out, heads, emha_tile, ff_ratio, p)
          for p in drop_probs),
        # normalize
        LayerNorm3d(dim_out),
    )


class Encoder(nn.ModuleList):
    def __init__(self,
                 dim_in: int,
                 depths: Sequence[int],
                 dims: Iterable[int],
                 heads: Iterable[int],
                 kernels: Iterable[int],
                 strides: Iterable[int],
                 emha_tiles: Iterable[int],
                 ff_ratios: Iterable[int],
                 drop_prob: float = .0):
        # create drop paths probabilities (one for each stage's block)
        drop_probs = torch.linspace(0, drop_prob, sum(depths)).split([*depths])
        super().__init__([
            encoder_stage(*args)
            for args in zip([dim_in, *dims], dims, kernels, strides,
                            drop_probs, emha_tiles, heads, ff_ratios)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        ys = []
        for stage in self:
            x = stage(x)
            ys.append(x)
        return ys


class MiniHead(nn.ModuleList):
    def __init__(self, dims_in: Sequence[int], dim_out: int,
                 scale_factors: Iterable[int]):
        super().__init__([
            nn.Sequential(
                nn.Conv3d(dim_in, dim_out, 1),
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode='trilinear',
                    align_corners=False,
                ),
            ) for dim_in, scale_factor in zip(dims_in, scale_factors)
        ])

    def forward(self, xs: Iterable[torch.Tensor]) -> torch.Tensor:
        ys = [up(x) for up, x in zip(self, xs)]
        return _all_sum(*ys)


class Head(nn.Module):
    def __init__(self,
                 dims_in: Iterable[int],
                 dim: int,
                 dim_out: int,
                 scale_factors: Iterable[int],
                 dropout_ratio: float = 0.0):
        super().__init__()
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim_in, dim, 1, bias=False),
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode='trilinear',
                    align_corners=False,
                ),
            ) for dim_in, scale_factor in zip(dims_in, scale_factors)
        ])
        self.to_out = nn.Sequential(
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Conv3d(dim, dim_out, 1),
        )

    def forward(self, xs: Iterable[torch.Tensor]) -> torch.Tensor:
        ys = [up(x) for up, x in zip(self.ups, xs)]
        y = _all_sum(*ys)
        return self.to_out(y)


class SegFormer3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        *,
        depths: Sequence[int] = (2, 2, 2, 2),
        drop_prob: float = 0.0,
        # Poolings
        kernels: Sequence[int] = (7, 3, 3, 3),
        strides: Sequence[int] = (4, 2, 2, 2),
        # Width
        head_dim: int = 32,
        heads: Sequence[int] = (1, 2, 5, 8),
        emha_tiles: Sequence[int] = (8, 4, 2, 1),
        ff_ratios: Sequence[int] = (8, 8, 4, 4),
        # Decoder
        decoder_dim: int | None = 256,
        scale_factor: int | None = 1,
        dropout_ratio: float = 0.0,
    ):
        super().__init__()

        dims = [head_dim * heads_ for heads_ in heads]

        if scale_factor is None:
            scale_factor = strides[0]
        scale_factors = (torch.as_tensor([1, *strides[1:]]).cumprod(0)
                         * scale_factor).tolist()

        self.encoder = Encoder(
            channels,
            depths=depths,
            dims=dims,
            heads=heads,
            kernels=kernels,
            strides=strides,
            emha_tiles=emha_tiles,
            ff_ratios=ff_ratios,
            drop_prob=drop_prob,
        )
        self.decoder = (
            MiniHead(dims, num_classes, scale_factors)
            if decoder_dim is None else Head(dims, decoder_dim, num_classes,
                                             scale_factors, dropout_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


def segformer_b0(channels: int, num_classes: int) -> nn.Module:
    return SegFormer3D(
        channels,
        num_classes,
        depths=[2, 2, 2, 2],
        head_dim=32,
        ff_ratios=[8, 8, 4, 4],
        decoder_dim=256,
    )


def segformer_b1(channels: int, num_classes: int) -> nn.Module:
    return SegFormer3D(
        channels,
        num_classes,
        depths=[2, 2, 2, 2],
        head_dim=64,  # B0 x2
        ff_ratios=[8, 8, 4, 4],
        decoder_dim=256,
    )


def segformer_b2(channels: int, num_classes: int) -> nn.Module:
    return SegFormer3D(
        channels,
        num_classes,
        depths=[3, 3, 6, 3],  # B1 +[1, 1, 4, 1]
        head_dim=64,
        ff_ratios=[8, 8, 4, 4],
        decoder_dim=768,  # B1 x3
    )


def segformer_b3(channels: int, num_classes: int) -> nn.Module:
    return SegFormer3D(
        channels,
        num_classes,
        depths=[3, 3, 18, 3],  # B2 +[0, 0, 12, 0]
        head_dim=64,
        ff_ratios=[8, 8, 4, 4],
        decoder_dim=768,
    )


def segformer_b4(channels: int, num_classes: int) -> nn.Module:
    return SegFormer3D(
        channels,
        num_classes,
        depths=[3, 8, 27, 3],  # B3 +[0, 5, 9, 0]
        head_dim=64,
        ff_ratios=[8, 8, 4, 4],
        decoder_dim=768,
    )


def segformer_b5(channels: int, num_classes: int) -> nn.Module:
    return SegFormer3D(
        channels,
        num_classes,
        depths=[3, 6, 40, 3],  # B4 +[0, -2, +13, 0]
        head_dim=64,
        ff_ratios=[4, 4, 4, 4],  # B4 -[4, 4, 0, 0]
        decoder_dim=768,
    )
