import torch
import torch.nn as nn


def double_conv(cin: int, cout: int, mid: int | None = None) -> nn.Sequential:
    mid = mid or cout
    return nn.Sequential(
        nn.Conv3d(cin, mid, 3, padding=1, bias=False),
        nn.BatchNorm3d(mid),
        nn.ReLU(inplace=True),
        nn.Conv3d(mid, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


def maxpool_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.MaxPool3d(2),
        double_conv(cin, cout),
    )


class Up(nn.Module):
    def __init__(self, cin: int, cout: int, upsample: bool = True):
        super().__init__()
        self.up: nn.Module
        if upsample:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = double_conv(cin, cout, cin // 2)
        else:
            self.up = nn.ConvTranspose3d(cin, cin // 2, 2, stride=2)
            self.conv = double_conv(cin, cout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 scale: float = 1,
                 upsample: bool = True):
        super().__init__()

        dims = [int(c * scale) for c in (32, 64, 128, 256, 512)]
        factor = 2 if upsample else 1

        self.stem = double_conv(num_channels, dims[0])
        self.enc = nn.ModuleList([
            maxpool_conv(dims[0], dims[1]),
            maxpool_conv(dims[1], dims[2]),
            maxpool_conv(dims[2], dims[3]),
            maxpool_conv(dims[3], dims[4] // factor),
        ])

        self.dec = nn.ModuleList([
            Up(dims[4], dims[3] // factor, upsample),
            Up(dims[3], dims[2] // factor, upsample),
            Up(dims[2], dims[1] // factor, upsample),
            Up(dims[1], dims[0], upsample),
        ])

        self.out = nn.Conv3d(dims[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        xs = [x]
        for enc in self.enc:
            x = enc(x)
            xs.append(x)

        x = xs.pop()
        for dec in self.dec:
            x = dec(x, xs.pop())

        return self.out(x)
