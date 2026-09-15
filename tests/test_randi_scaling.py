"""Checks for explicit Randi signal-scale policies."""

import numpy as np
import pytest

from brain_moe_pinn.data.ingest_randi import _clean_calcium


def test_prefix_minmax_does_not_use_future_frames():
    raw = np.array([[1.0], [2.0], [100.0]], dtype=np.float64)
    cleaned, mask, stats = _clean_calcium(
        raw,
        min_value=0.0,
        max_value=None,
        scale="prefix_minmax",
        fit_frames=2,
    )
    assert mask[0].tolist() == [True, True, True]
    assert np.allclose(cleaned[0], [-3.0, 3.0, 591.0])
    assert stats["scale_fit_frames"] == 2
    assert stats["scale_fit_uses_future_frames"] == 0


def test_legacy_minmax_records_that_it_uses_full_recording():
    raw = np.array([[1.0], [2.0], [100.0]], dtype=np.float64)
    cleaned, _, stats = _clean_calcium(
        raw,
        min_value=0.0,
        max_value=None,
        scale="minmax",
    )
    assert np.allclose(cleaned[0], [-3.0, -2.9393939, 3.0], atol=1e-6)
    assert stats["scale_fit_frames"] == 3
    assert stats["scale_fit_uses_future_frames"] == 1


def test_raw_mode_preserves_values_and_requires_no_hidden_scale():
    raw = np.array([[5.0, 10.0], [6.0, 11.0]], dtype=np.float64)
    cleaned, mask, stats = _clean_calcium(
        raw,
        min_value=0.0,
        max_value=None,
        scale="none",
    )
    assert np.array_equal(cleaned, raw.T.astype(np.float32))
    assert mask.all()
    assert stats["scale_fit_frames"] == 0
    assert stats["scale_fit_uses_future_frames"] == 0


def test_prefix_scale_requires_explicit_fit_window():
    with pytest.raises(ValueError, match="requires fit_frames"):
        _clean_calcium(
            np.ones((3, 1)),
            min_value=0.0,
            max_value=None,
            scale="prefix_minmax",
        )
