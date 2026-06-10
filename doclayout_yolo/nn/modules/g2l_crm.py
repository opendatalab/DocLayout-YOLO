import math
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

from .conv import Conv
from .block import CIB

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DilatedBlock(nn.Module):
    """Standard bottleneck with dilated convolution."""

    def __init__(self, c, dilation, k, fuse="sum", shortcut=True):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        self.dilation = dilation
        self.k = k
        self.cv2 = Conv(c, c, k=1, s=1)
        self.add = shortcut

        self.fuse = fuse
        if fuse == "glu":
            self.conv_gating = Conv(c*len(self.dilation), c*len(self.dilation), k=1, s=1, g=c*len(self.dilation))
            self.conv1x1 = Conv(c*len(self.dilation), c, k=1, s=1, g=c)
        elif fuse == "sum":
            self.conv1x1 = Conv(c, c, k=1, s=1, g=c)

        self.dcv = Conv(c, c, k=self.k, s=1)

    def dilated_conv(self, x, dilation):
        """Apply dilated convolution using the shared Conv weight.

        Handles fused checkpoints where `bn` may be missing or replaced with nn.Identity.
        """
        weight = self.dcv.conv.weight
        padding = dilation * (self.k // 2)

        # Check if bn exists and is a real BatchNorm (not Identity)
        bn = getattr(self.dcv, "bn", None)
        has_bn = isinstance(bn, nn.BatchNorm2d)

        if has_bn:
            x = F.conv2d(x, weight, stride=1, padding=padding, dilation=dilation)
            x = bn(x)
        else:
            # Fused: use bias directly from conv
            bias = getattr(self.dcv.conv, "bias", None)
            x = F.conv2d(x, weight, bias=bias, stride=1, padding=padding, dilation=dilation)

        return self.dcv.act(x)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        dx = [self.dilated_conv(x, d) for d in self.dilation]
        dx = [self.cv2(_dx) for _dx in dx]
        if self.fuse == "glu":
            dx = torch.cat(dx, dim=1)
            G = torch.sigmoid(self.conv_gating(dx))
            dx = dx * G  # Element-wise multiplication
            dx = self.conv1x1(dx)
        elif self.fuse == "sum":
            # Optimized: avoid torch.stack allocation of [n_dilation, B, C, H, W]
            dx = sum(dx)
            dx = self.conv1x1(dx)

        return x + dx if self.add else dx

    def fuse_custom(self):
        """Fuse Conv+BN in all submodules."""
        for m in [self.cv2, self.conv1x1, self.dcv]:
            if hasattr(m, "fuse_custom") and not getattr(m, "fused", False):
                m.fuse_custom()
        return self


class DilatedBottleneck(nn.Module):
    """Standard bottleneck with dilated convolution."""

    def __init__(self, c1, c2, shortcut=True, dilation=[1,2,3], block_k=3, fuse="sum", g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()

        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        self.dilated_block = DilatedBlock(c_, dilation, block_k, fuse)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.dilated_block(self.cv1(x))) if self.add else self.cv2(self.dilated_block(self.cv1(x)))

    def fuse_custom(self):
        """Fuse Conv+BN in all submodules."""
        for m in [self.cv1, self.cv2]:
            if hasattr(m, "fuse_custom") and not getattr(m, "fused", False):
                m.fuse_custom()
        if hasattr(self.dilated_block, "fuse_custom"):
            self.dilated_block.fuse_custom()
        return self


class G2L_CRM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, use_dilated=False, dilation=[1,2,3], block_k=3, fuse="sum", g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if use_dilated:
            self.m = nn.ModuleList(DilatedBottleneck(
                self.c, 
                self.c, 
                shortcut, 
                dilation, 
                block_k, 
                fuse, 
                g, 
                k=((3, 3), (3, 3)), 
                e=1.0) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(CIB(
                self.c, 
                self.c, 
                shortcut, 
                e=1.0) for _ in range(n)
            )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def fuse_custom(self):
        """Fuse all Conv+BN in G2L_CRM."""
        for m in [self.cv1, self.cv2]:
            if hasattr(m, "fuse_custom") and not getattr(m, "fused", False):
                m.fuse_custom()
        for m in self.m:
            if hasattr(m, "fuse_custom"):
                m.fuse_custom()
        return self