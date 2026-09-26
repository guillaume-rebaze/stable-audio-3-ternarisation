from __future__ import annotations

import random
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import train_ternary_quality as tq  # noqa: E402
import training_checkpoint as tc  # noqa: E402
from train_ternary_window_v6 import (  # noqa: E402
    optimizer_learning_rate,
    paired_epoch_pair_index,
    paired_epoch_trajectory_coverage,
    validate_warm_start_payload,
)


def _checkpoint_fixture(root: Path):
    run_dir = root / "parent-run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    records = run_dir / "records_checkpoint.npz"
    records.write_bytes(b"fixed-ternary-records")
    teacher = root / "teacher.npz"
    teacher.write_bytes(b"fixed-teacher-identity")

    model = nn.Linear(4, 3)
    optimizer = optim.AdamW(1e-3, eps=1e-6, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())
    x = mx.ones((2, 4), dtype=mx.float32)
    target = mx.zeros((2, 3), dtype=mx.float32)
    loss_fn = nn.value_and_grad(
        model,
        lambda current, values, expected: mx.mean(
            (current(values) - expected) ** 2
        ),
    )
    loss, grads = loss_fn(model, x, target)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    checkpoint = checkpoint_dir / "window_latest.npz"
    tc.save_step_checkpoint(
        checkpoint,
        model.trainable_parameters(),
        {"main": optimizer.state},
        {
            "window": [0, 1],
            "step_next": 1,
            "group_size": 32,
            "quantizer_mode": "symmetric",
            "run_signature": {
                "output_dir": str(run_dir.resolve()),
                "window": [0, 1],
                "group_size": 32,
                "quantizer_mode": "symmetric",
                "weight_decay": 0.0,
                "optimizer_eps": 1e-6,
                "gradient_accumulation": 4,
                "seed": 20260924,
                "teacher": tq.file_fingerprint(teacher),
            },
        },
        random.Random(20260924).getstate(),
        np.random.get_state(),
        list(getattr(mx.random, "state", [])),
    )
    loaded = tc.load_step_checkpoint(checkpoint)
    return model, optimizer, checkpoint, records, teacher, loaded


def test_warm_start_validates_fp32_masters_optimizer_and_parent_records(
    tmp_path: Path,
) -> None:
    model, _, checkpoint, records, teacher, state = _checkpoint_fixture(tmp_path)
    result = validate_warm_start_payload(
        checkpoint,
        state,
        model.trainable_parameters(),
        records,
        teacher,
        (0, 1),
        32,
        "symmetric",
        0.0,
        1e-6,
        4,
        20260924,
    )
    assert result["step_next"] == result["optimizer_step"] == 1
    assert result["master_parameter_keys"] == 2
    assert result["effective_learning_rate"] == pytest.approx(1e-3)


def test_warm_start_rejects_records_not_from_checkpoint_parent(
    tmp_path: Path,
) -> None:
    model, _, checkpoint, records, teacher, state = _checkpoint_fixture(tmp_path)
    other_records = tmp_path / "other-records.npz"
    other_records.write_bytes(b"different-records")
    with pytest.raises(ValueError, match="source records differ"):
        validate_warm_start_payload(
            checkpoint,
            state,
            model.trainable_parameters(),
            other_records,
            teacher,
            (0, 1),
            32,
            "symmetric",
            0.0,
            1e-6,
            4,
            20260924,
        )


def test_paired_epoch_uses_each_pair_once_for_trajectory_loss() -> None:
    pair_count = 224
    order = random.Random(20260924).sample(range(pair_count), pair_count)
    coverage = paired_epoch_trajectory_coverage(order, updates=112)
    assert coverage == set(range(pair_count))
    for update in range(112):
        assert paired_epoch_pair_index(order, update, 0) == paired_epoch_pair_index(
            order, update, 1
        )
        assert paired_epoch_pair_index(order, update, 2) == paired_epoch_pair_index(
            order, update, 3
        )


def test_fixed_learning_rate_survives_optimizer_state_transfer(tmp_path: Path) -> None:
    model, old_optimizer, checkpoint, records, teacher, state = _checkpoint_fixture(
        tmp_path
    )
    del old_optimizer, checkpoint, records, teacher
    model.update(state["model_state"])
    expected_rate = optimizer_learning_rate(
        type("OptimizerStateView", (), {"state": state["optimizer_state"]["main"]})()
    )
    resumed = optim.AdamW(
        learning_rate=optim.cosine_decay(expected_rate, 4, end=expected_rate),
        eps=1e-6,
        weight_decay=0.0,
    )
    resumed.init(model.trainable_parameters())
    resumed.state = state["optimizer_state"]["main"]
    loss_fn = nn.value_and_grad(
        model,
        lambda current, values, expected: mx.mean(
            (current(values) - expected) ** 2
        ),
    )
    loss, grads = loss_fn(
        model,
        mx.ones((2, 4), dtype=mx.float32),
        mx.zeros((2, 3), dtype=mx.float32),
    )
    resumed.update(model, grads)
    mx.eval(model.parameters(), resumed.state, loss)
    assert optimizer_learning_rate(resumed) == pytest.approx(expected_rate, rel=1e-7)
