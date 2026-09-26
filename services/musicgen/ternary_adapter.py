"""Small FP16/FP32 residual adapters around MLX affine ternary layers."""

from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn


class TernaryAdapterLinear(nn.Module):
    """Frozen affine ternary base plus a low-rank residual.

    The base remains the exact MLX QuantizedLinear contract.  The adapter is
    intentionally explicit: it is not folded back into the ternary weight.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        seed: int = 0,
    ):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        # MLX stores 2-bit weights as uint32, i.e. 16 packed codes/word.
        group_size = int(base.weight.shape[1] * 16 // base.scales.shape[1])
        in_dim = int(base.weight.shape[1] * 16)
        out_dim = int(base.weight.shape[0])
        scale = 1.0 / np.sqrt(max(in_dim, 1))
        self.down = mx.random.normal(
            (self.rank, in_dim), dtype=mx.float32, key=mx.random.key(seed)
        ) * scale
        self.up = mx.zeros((out_dim, self.rank), dtype=mx.float32)
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        base_output = self.base(x)
        residual = x.astype(mx.float32) @ self.down.T
        residual = residual @ self.up.T
        residual = residual * (self.alpha / max(self.rank, 1))
        return base_output + residual.astype(base_output.dtype)
