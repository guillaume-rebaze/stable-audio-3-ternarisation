import math

import numpy as np
import soundfile as sf

from run_sftberlin_quantized import DEFAULT_GROUP_SIZE, validate_quantization
from quantize_sftberlin import compare_audio, _runtime_observations, _variant_name


def test_quantization_validation_accepts_int1_int2_int4_int8_and_group_multiples_of_32():
    assert validate_quantization(None) == (None, DEFAULT_GROUP_SIZE)
    assert validate_quantization(8) == (8, DEFAULT_GROUP_SIZE)
    assert validate_quantization(4, 32) == (4, 32)
    assert validate_quantization(2) == (2, DEFAULT_GROUP_SIZE)
    assert validate_quantization(1) == (1, DEFAULT_GROUP_SIZE)

    for bits in (3, 16):
        try:
            validate_quantization(bits)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid bits accepted: {bits}")
    for group_size in (0, 31, 48):
        try:
            validate_quantization(4, group_size)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid group size accepted: {group_size}")


def test_variant_names_and_runtime_observations_are_stable():
    assert [_variant_name(value) for value in ("FP16", "int8", "INT4")] == ["fp16", "int8", "int4"]
    assert _variant_name("int2") == "int2"
    assert _variant_name("int1") == "int1"

    observations = _runtime_observations(
        "sample 1234 ms  (154 ms/step)\n    DiT sample  █  2.75 GB\n"
    )
    assert observations["sample_ms"] == 1234.0
    assert observations["sample_ms_per_step"] == 154.0
    assert observations["peak_ram_gb_reported"] == 2.75


def test_compare_audio_reports_matched_difference(tmp_path):
    rate = 44_100
    axis = np.arange(rate, dtype=np.float32) / rate
    reference = np.column_stack((0.2 * np.sin(2 * math.pi * 110 * axis),) * 2)
    candidate = reference * 0.5
    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    sf.write(reference_path, reference, rate, subtype="FLOAT")
    sf.write(candidate_path, candidate, rate, subtype="FLOAT")

    report = compare_audio(reference_path, candidate_path)

    assert report["correlation"] > 0.999999
    assert report["reference_frames"] == rate
    assert report["candidate_frames"] == rate
    assert report["rms_delta_db"] < -5.9
    assert report["diff_rms_db_relative"] < -5.9
