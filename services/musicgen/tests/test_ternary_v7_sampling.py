from pathlib import Path
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parents[1]))

from build_ternary_state_cache import (  # noqa: E402
    genre_family,
    model_rollout_states,
    stratified_prompt_subset,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    build_pingpong_schedule,
    sample_flow_pingpong,
)
from train_ternary_window_v6 import (  # noqa: E402
    balanced_fixed_state_indices,
    cosine_schedule_value,
    tree_is_finite,
)
from ternary_runtime_contract import pingpong_trace  # noqa: E402


def test_prompt_subset_round_robins_genre_labels() -> None:
    samples = [
        {"prompt": f"{genre}-{index}", "genre": genre}
        for genre in ("ambient", "electronic", "rock")
        for index in range(3)
    ]
    selected = stratified_prompt_subset(samples, 6)
    assert len(selected) == 6
    assert sum(prompt.startswith("ambient-") for prompt in selected) == 2
    assert sum(prompt.startswith("electronic-") for prompt in selected) == 2
    assert sum(prompt.startswith("rock-") for prompt in selected) == 2


def test_genre_family_merges_sft_subgenres_but_preserves_music_families() -> None:
    assert genre_family("aphex") == "electronic"
    assert genre_family("house") == "electronic"
    assert genre_family("ambient_cinematic") == "ambient"
    assert genre_family("voice, a-cappella, live") == "vocal"
    assert genre_family("classical_piano") == "classical"


def test_fixed_micro_overfit_subset_balances_prompt_timestep_and_source() -> None:
    prompts = np.repeat(np.arange(8, dtype=np.int32), 6)
    sigmas = np.tile(np.repeat(np.array([0.95, 0.5, 0.1], dtype=np.float32), 2), 8)
    sources = np.tile(np.array(["real_latent_noised", "teacher_trajectory"]), 24)
    indices = balanced_fixed_state_indices(prompts, sigmas, sources, 16)

    assert len(set(indices)) == 16
    assert len(set(prompts[indices])) == 4
    assert set(sources[indices]) == {"real_latent_noised", "teacher_trajectory"}
    assert len(set(sigmas[indices])) == 2


def test_training_numerics_checks_nonfinite_gradients_and_schedule_endpoints() -> None:
    assert tree_is_finite({"weight": mx.array([1.0, -2.0], dtype=mx.float32)})
    assert not tree_is_finite({"weight": mx.array([1.0, float("nan")])})
    assert cosine_schedule_value(1e-4, 1e-6, 50, 0) == 1e-4
    assert cosine_schedule_value(1e-4, 1e-6, 50, 50) == 1e-6


def test_recorded_pingpong_trace_matches_production_rng_and_transitions() -> None:
    generation_seed = 4242
    sampler_seed = generation_seed + 1
    sigmas = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    initial = mx.random.normal(
        (1, 4, 16), dtype=mx.float16, key=mx.random.key(generation_seed)
    )
    production_inputs: list[np.ndarray] = []

    def production_model(x: mx.array, t: mx.array) -> mx.array:
        assert t.dtype == mx.float32
        production_inputs.append(np.asarray(x).astype(np.float32))
        return mx.zeros_like(x)

    expected_terminal = sample_flow_pingpong(
        production_model, initial, sigmas, seed=sampler_seed
    )
    trace = pingpong_trace(
        lambda x, _t: mx.zeros_like(x), initial, sigmas, sampler_seed=sampler_seed
    )

    assert len(trace) == len(production_inputs) + 1
    for index, production_state in enumerate(production_inputs):
        np.testing.assert_array_equal(trace[index]["state"], production_state)
    np.testing.assert_array_equal(
        trace[-1]["state"], np.asarray(expected_terminal).astype(np.float32)
    )
    assert trace[-2]["noise"] is None


def test_teacher_rollout_cache_preserves_batch_axis_for_stack() -> None:
    def zero_model(x: mx.array, _t: mx.array, _cross: mx.array, _global: mx.array):
        return mx.zeros_like(x)

    states = model_rollout_states(
        zero_model,
        mx.zeros((1, 1, 1), dtype=mx.float16),
        mx.zeros((1, 1), dtype=mx.float16),
        latent_len=4,
        steps=2,
        seed=9,
    )

    assert len(states) == 2
    assert all(state.shape == (1, 256, 4) for state, _sigma in states)
