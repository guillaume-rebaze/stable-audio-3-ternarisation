from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parents[1]))

import training_checkpoint as tc  # noqa: E402


def _assert_tree_equal(left, right) -> None:
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    assert set(left_flat) == set(right_flat)
    for key in left_flat:
        assert np.array_equal(np.asarray(left_flat[key]), np.asarray(right_flat[key]))
        assert np.asarray(left_flat[key]).dtype == np.asarray(right_flat[key]).dtype


def test_step_checkpoint_roundtrip_preserves_optimizer_and_rng(tmp_path) -> None:
    model = nn.Linear(4, 3)
    optimizer = optim.AdamW(1e-3, eps=1e-4)
    optimizer.init(model.trainable_parameters())
    x = mx.ones((2, 4), dtype=mx.float32)
    target = mx.zeros((2, 3), dtype=mx.float32)

    def loss_fn(current, values, expected):
        prediction = current(values)
        return mx.mean((prediction - expected) ** 2)

    value_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = value_grad(model, x, target)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    py_rng = random.Random(12)
    py_rng.random()
    numpy_rng = np.random.RandomState(34)
    numpy_rng.rand()

    path = tmp_path / "step.npz"
    tc.save_step_checkpoint(
        path,
        model.trainable_parameters(),
        optimizer.state,
        {"block_index": 0, "step_next": 1, "group_size": 64, "crop_len": 128, "quantizer_mode": "symmetric"},
        py_rng.getstate(),
        numpy_rng.get_state(),
        list(getattr(mx.random, "state", [])),
    )
    loaded = tc.load_step_checkpoint(path)
    _assert_tree_equal(model.trainable_parameters(), loaded["model_state"])
    _assert_tree_equal(optimizer.state, loaded["optimizer_state"])
    assert loaded["metadata"]["step_next"] == 1

    restored = tc.restore_rngs(loaded)
    assert restored.getstate() == py_rng.getstate()


def test_step_checkpoint_rejects_modified_payload(tmp_path) -> None:
    py_rng = random.Random(5)
    path = tmp_path / "step.npz"
    tc.save_step_checkpoint(
        path,
        {"weight": mx.ones((2,), dtype=mx.float32)},
        {"step": mx.array(1, dtype=mx.int32)},
        {"step_next": 1},
        py_rng.getstate(),
        np.random.get_state(),
    )
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        tc.load_step_checkpoint(path)


def test_optimizer_and_rng_resume_matches_continuous_run_in_new_process(tmp_path) -> None:
    initial_weight = np.arange(8, dtype=np.float32).reshape(2, 4) / 10
    initial_bias = np.array([0.05, -0.1], dtype=np.float32)
    inputs = np.arange(40, dtype=np.float32).reshape(10, 4) / 20
    targets = np.arange(20, dtype=np.float32).reshape(10, 2) / 30
    checkpoint = tmp_path / "resume.npz"
    expected_path = tmp_path / "continuous.npz"
    resumed_path = tmp_path / "resumed.npz"

    model = nn.Linear(4, 2)
    model.weight = mx.array(initial_weight)
    model.bias = mx.array(initial_bias)
    optimizer = optim.AdamW(1e-3, eps=1e-6)
    optimizer.init(model.trainable_parameters())
    rng = random.Random(4242)
    value_grad = nn.value_and_grad(
        model,
        lambda current, x, target: mx.mean((current(x) - target) ** 2),
    )
    for step in range(8):
        index = rng.randrange(len(inputs))
        loss, grads = value_grad(
            model,
            mx.array(inputs[index:index + 1]),
            mx.array(targets[index:index + 1]),
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 3:
            tc.save_step_checkpoint(
                checkpoint,
                model.trainable_parameters(),
                optimizer.state,
                {"step_next": 4, "run_signature": {"seed": 4242}},
                rng.getstate(),
                np.random.get_state(),
                list(getattr(mx.random, "state", [])),
            )
    np.savez(expected_path, **{key: np.asarray(value) for key, value in model.parameters().items()})

    worker = r"""
import random, sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import training_checkpoint as tc

checkpoint, output = Path(sys.argv[1]), Path(sys.argv[2])
weight = np.arange(8, dtype=np.float32).reshape(2, 4) / 10
bias = np.array([0.05, -0.1], dtype=np.float32)
inputs = np.arange(40, dtype=np.float32).reshape(10, 4) / 20
targets = np.arange(20, dtype=np.float32).reshape(10, 2) / 30
model = nn.Linear(4, 2)
model.weight, model.bias = mx.array(weight), mx.array(bias)
optimizer = optim.AdamW(1e-3, eps=1e-6)
optimizer.init(model.trainable_parameters())
state = tc.load_step_checkpoint(checkpoint)
model.update(state["model_state"])
optimizer.state = state["optimizer_state"]
rng = tc.restore_rngs(state, random.Random(4242))
value_grad = nn.value_and_grad(
    model, lambda current, x, target: mx.mean((current(x) - target) ** 2)
)
for step in range(state["metadata"]["step_next"], 8):
    index = rng.randrange(len(inputs))
    loss, grads = value_grad(
        model, mx.array(inputs[index:index + 1]), mx.array(targets[index:index + 1])
    )
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
np.savez(output, **{key: np.asarray(value) for key, value in model.parameters().items()})
"""
    subprocess.run(
        [sys.executable, "-c", worker, str(checkpoint), str(resumed_path)],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(expected_path, allow_pickle=False) as expected, np.load(
        resumed_path, allow_pickle=False
    ) as resumed:
        for key in expected.files:
            np.testing.assert_allclose(resumed[key], expected[key], rtol=1e-7, atol=1e-7)
