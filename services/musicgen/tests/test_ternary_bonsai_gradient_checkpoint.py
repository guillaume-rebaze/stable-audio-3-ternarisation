from __future__ import annotations

from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

sys.path.insert(0, str(Path(__file__).parents[1]))


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 7)
        self.linear2 = nn.Linear(7, 4)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(mx.tanh(self.linear1(x)))


def test_gradient_checkpoint_matches_dense_gradient_and_update() -> None:
    mx.random.seed(20260925)
    dense = TinyBlock()
    checkpointed = TinyBlock()
    checkpointed.update(dense.parameters())
    x = mx.arange(12, dtype=mx.float32).reshape(3, 4) / 10.0
    target = mx.full((3, 4), 0.25, dtype=mx.float32)

    def loss(model: TinyBlock, values: mx.array) -> mx.array:
        prediction = model(values)
        return mx.mean((prediction - target) ** 2)

    def checkpointed_loss(model: TinyBlock, values: mx.array) -> mx.array:
        def recompute(params: object, inner_values: mx.array) -> mx.array:
            model.update(params)
            return model(inner_values)

        prediction = mx.checkpoint(recompute)(model.trainable_parameters(), values)
        return mx.mean((prediction - target) ** 2)

    dense_value_grad = nn.value_and_grad(dense, loss)
    checkpointed_value_grad = nn.value_and_grad(checkpointed, checkpointed_loss)
    dense_loss, dense_grads = dense_value_grad(dense, x)
    checkpointed_loss_value, checkpointed_grads = checkpointed_value_grad(checkpointed, x)
    mx.eval(dense_loss, dense_grads, checkpointed_loss_value, checkpointed_grads)
    assert float(dense_loss) == float(checkpointed_loss_value)

    dense_flat = [(key, np.asarray(value)) for key, value in tree_flatten(dense_grads)]
    checkpointed_flat = [
        (key, np.asarray(value)) for key, value in tree_flatten(checkpointed_grads)
    ]
    assert [key for key, _ in dense_flat] == [key for key, _ in checkpointed_flat]
    for (_, left), (_, right) in zip(dense_flat, checkpointed_flat):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)

    def apply_update(model: TinyBlock, gradients: object) -> None:
        model.update(
            tree_map(
                lambda parameter, gradient: parameter - 0.01 * gradient,
                model.parameters(),
                gradients,
            )
        )

    apply_update(dense, dense_grads)
    apply_update(checkpointed, checkpointed_grads)
    dense_after = dense(x)
    checkpointed_after = checkpointed(x)
    mx.eval(dense_after, checkpointed_after)
    np.testing.assert_allclose(
        np.asarray(dense_after), np.asarray(checkpointed_after), rtol=1e-6, atol=1e-6
    )
