"""ZAPBench adapter (zebrafish) regressions.

Covers the multi-species contract: a second animal with a completely different
clock (914 ms volumes), channel space (71,721 cells -> region reduction),
reporter (nuclear GCaMP7f) and control width (26 stimulus features) must land in
the same canonical ladder without borrowing anything from the C. elegans stage.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from brain_moe_pinn.data.species_dataset import (
    SpeciesSignalDataset, build_species_dataloaders,
)
from brain_moe_pinn.data.zapbench import (
    CONDITION_NAMES, CONDITION_OFFSETS, DEFAULT_RATE_HZ, condition_spans, ingest,
    region_bins, select_channels,
)


def _write_fixture(path: Path, n_steps: int = 120, n_cells: int = 12,
                   n_features: int = 4) -> Path:
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((n_steps, n_cells)).astype("float32")
    stimuli = rng.standard_normal((n_steps, n_features)).astype("float32")
    np.savez(path, traces=traces, stimuli=stimuli)
    return path


def test_condition_spans_follow_the_release_layout():
    spans = condition_spans()
    assert [s.name for s in spans] == list(CONDITION_NAMES)
    assert len(spans) == len(CONDITION_NAMES)
    # 1-step padding inside each condition boundary
    assert spans[0].start == CONDITION_OFFSETS[0] + 1
    assert spans[0].stop == CONDITION_OFFSETS[1] - 1
    assert spans[-1].stop == CONDITION_OFFSETS[-1] - 1
    with pytest.raises(ValueError, match="offsets"):
        condition_spans(offsets=(0, 10, 20), names=("a",))


def test_ingest_writes_sessions_control_and_provenance(tmp_path):
    source = _write_fixture(tmp_path / "traces.npz")
    out = tmp_path / "ladder"
    ingest(source, out, rate_hz=DEFAULT_RATE_HZ, conditions="a,b",
           condition_offsets=(0, 40, 80), condition_names=("a", "b"),
           condition_padding=0, max_cells=5, overwrite=True)
    import csv
    rows = {r["sample_id"]: r for r in
            csv.DictReader(open(out / "manifest.csv"))}
    assert set(rows) == {"00_a", "01_b"}
    row = rows["00_a"]
    assert row["species"] == "zebrafish"
    assert row["imaging"] == "lsfm_spim"
    assert row["reporter"] == "h2b_gcamp7f"
    assert float(row["rate_hz"]) == pytest.approx(DEFAULT_RATE_HZ)
    assert float(row["dt_s"]) == pytest.approx(0.914, abs=1e-3)
    assert row["subject"] == row["animal"]           # one animal, many sessions
    # one sample per condition: session carries the protocol name
    assert row["session"] == "a" and row["condition"] == "a"

    calcium = np.load(out / "calcium" / "00_a.npy")
    control = np.load(out / "stimulus" / "00_a.npy")
    assert calcium.shape == (5, 40)                  # (C, T), reduced channels
    assert control.shape == (4, 40)                  # 26-d bank in the release
    assert len(calcium) != len(control)              # control is not a signal


def test_ingest_requires_an_explicit_rate(tmp_path):
    source = _write_fixture(tmp_path / "traces.npz")
    with pytest.raises(ValueError, match="rate-hz"):
        ingest(source, tmp_path / "ladder")


def test_channel_reduction_keeps_the_high_variance_cells():
    rng = np.random.default_rng(1)
    traces = rng.standard_normal((50, 6)).astype("float32")
    traces[:, 3] *= 20.0
    keep = select_channels(traces, max_cells=2)
    assert len(keep) == 2 and 3 in keep
    assert len(select_channels(traces, max_cells=99)) == 6


def test_region_bins_pool_cells_into_spatial_grid():
    positions = {
        "x": np.array([0.0, 1.0, 10.0, 11.0]),
        "y": np.array([0.0, 0.0, 0.0, 0.0]),
        "z": np.array([0.0, 0.0, 0.0, 0.0]),
    }
    assignment, labels = region_bins(positions, (1, 1, 2), n_cells=4)
    assert len(labels) == 2 and set(assignment) == {0, 1}
    from brain_moe_pinn.data.zapbench import pool_by_bin

    traces = np.stack([np.ones(3), np.ones(3), np.zeros(3), np.zeros(3)], axis=1)
    pooled = pool_by_bin(traces.astype("float32"), assignment, len(labels))
    assert pooled.shape == (2, 3)
    assert pooled[0].tolist() == [1.0, 1.0, 1.0]
    assert pooled[1].tolist() == [0.0, 0.0, 0.0]


def test_ladder_loads_with_condition_split_and_manifest_clock(tmp_path):
    source = _write_fixture(tmp_path / "traces.npz", n_steps=240, n_cells=8)
    out = tmp_path / "ladder"
    ingest(source, out, rate_hz=1.0938, conditions="a,b,c",
           condition_offsets=(0, 80, 160, 240), condition_names=("a", "b", "c"),
           condition_padding=0, max_cells=4, overwrite=True)
    ds = SpeciesSignalDataset(out, ["calcium", "stimulus"], seq_len=16,
                              roles={"calcium": "signal", "stimulus": "control"},
                              species="zebrafish", return_next_step_targets=True,
                              random_windows=False)
    item = ds[0]
    assert item["dt"] == pytest.approx(1.0 / 1.0938, rel=1e-4)
    assert item["species"] == "zebrafish"
    assert item["calcium"].shape == (4, 16)
    assert item["stimulus"].shape[0] == 4          # control track, its own width

    loaders = build_species_dataloaders(
        out, ["calcium", "stimulus"], batch_size=1, seq_len=16, num_workers=0,
        shuffle=False, split_by="condition", val_frac=0.34,
        roles={"calcium": "signal", "stimulus": "control"},
        species="zebrafish", return_next_step_targets=True)
    train_loader, val_loader = loaders[:2]
    val_conditions = {row["condition"] for row in val_loader.dataset.rows}
    train_conditions = {row["condition"] for row in train_loader.dataset.rows}
    assert val_conditions and not (val_conditions & train_conditions)
    batch = next(iter(train_loader))
    assert batch["species"] == ["zebrafish"]
