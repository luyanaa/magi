"""C. elegans data-path regressions.

Every test here pins a defect found in the 2026-09 audit:

* per-sample imaging rates (3.69-5.72 fps) instead of a silent 10 Hz default,
  and the salt drive/trial windows derived from them;
* the reference channel quality filter and the ``--apply-names`` +
  ``--filter-known`` interaction that used to drop every channel;
* fail-loud behaviour for rate mismatches, empty selections and re-ingest;
* the reconstruction head giving channels *distinct waveforms* (the old
  parameterisation was identical up to a DC offset);
* observation-channel statistics/pooling that respect validity masks;
* per-sample ``dt`` and species tags reaching the model;
* the CBM-style parameter estimators (tau, coupling split).
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from brain_moe_pinn.data.readers import _write_rows, emit_sample
from brain_moe_pinn.data.species_dataset import (
    SpeciesSignalDataset, build_species_dataloaders,
)
from brain_moe_pinn.data.stimulus import (
    read_stimulation_timing, salt_waveform, trial_windows,
)


# --------------------------------------------------------------------------- #
# fixtures: a miniature salt corpus in the released format
# --------------------------------------------------------------------------- #
def _write_worm(directory: Path, index: int, n_frames: int, n_channels: int,
                seed: int = 0, nan_cells=()) -> None:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_frames, n_channels))
    for row, col in nan_cells:
        data[row, col] = np.nan
    directory.mkdir(parents=True, exist_ok=True)
    lines = [",".join("" if np.isnan(v) else f"{v:.6f}" for v in row)
             for row in data]
    (directory / f"{index}_ratio.csv").write_text("\n".join(lines) + "\n")
    names = [f"N{index}_{c}" for c in range(n_channels)]
    (directory / f"{index}_uniqNames.csv").write_text("\n".join(names) + "\n")


def _write_timing_csv(path: Path, rows) -> None:
    header = ("#sample,#animal,name,anesthesia,frames/sec,"
              "frame number of first stimuli (NaCl),"
              "every N frames NaCl concentration change,"
              "number of stimulation cycles,total duration (sec)")
    body = ["\n".join(",".join(str(v) for v in row) for row in rows)]
    path.write_text(header + "\n" + body[0] + "\n")


def _metadata_dir(tmp_path: Path, names) -> Path:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "conneurons.csv").write_text("\n".join(names) + "\n")
    (meta / "globalNames.csv").write_text("\n".join(names) + "\n")
    return meta


# --------------------------------------------------------------------------- #
# A1/A2/A3: per-sample rates, stimulus track, animal identity
# --------------------------------------------------------------------------- #
def test_ingest_uses_per_sample_rates_and_writes_stimulus(tmp_path):
    from brain_moe_pinn.data.ingest_c_elegans import convert_directory

    raw = tmp_path / "raw"
    _write_worm(raw, 1, n_frames=400, n_channels=3, seed=1)
    _write_worm(raw, 2, n_frames=400, n_channels=3, seed=2)
    timing_path = tmp_path / "stimulation_timing.csv"
    _write_timing_csv(timing_path, [
        (1, 7, "a", 0, 5.0, 50.0, 25.0, 12.0, 80.0),
        (2, 9, "b", 1, 4.0, 20.0, 20.0, 15.0, 100.0),
    ])
    timing = read_stimulation_timing(timing_path)
    out = tmp_path / "ladder"
    convert_directory(raw, out, timing=timing, metadata_dir=_metadata_dir(
        tmp_path, [f"N{i}_{c}" for i in (1, 2) for c in range(3)]),
        filter_known=True, qc_autocorr=None, stimulus_track=True)

    manifest = {row["sample_id"]: row
                for row in _read_manifest(out / "manifest.csv")}
    assert manifest["1"]["rate_hz"] == "5.0"
    assert manifest["2"]["rate_hz"] == "4.0"
    assert float(manifest["2"]["dt_s"]) == pytest.approx(0.25)
    # subject is the animal, so a re-imaged animal cannot be split across sets
    assert manifest["1"]["subject"] == "7"
    assert manifest["2"]["subject"] == "9"
    assert manifest["2"]["anesthesia"] == "1"
    assert manifest["1"]["sample_name"] == "a"

    drive = np.load(out / "stimulus" / "1.npy")
    assert drive.shape == (1, 400)
    assert np.all(drive[0, :50] == 0)          # pre-stimulus span is exactly zero
    assert np.any(drive[0, 50:] != 0)
    trials = json.loads((out / "stimulus_trials" / "1.json").read_text())
    assert trials[0][0] == 50
    assert all(b == c for (_, b), (c, _) in zip(trials, trials[1:]))  # no gaps


def test_salt_waveform_matches_reference_shape():
    n, start, period = 200, 10.0, 20.0
    y = salt_waveform(n, start, period)
    assert np.all(y[:10] == 0)
    x = np.arange(1, n + 1) - start
    phase = np.sin(x / period * np.pi)
    expected = np.sign(phase) * np.abs(phase) ** 0.25
    expected[:10] = 0.0
    assert np.allclose(y, expected.astype(np.float32), atol=1e-6)
    windows = trial_windows(n, start, period)
    assert windows[0][0] == 10
    assert all(b == c for (_, b), (c, _) in zip(windows, windows[1:]))


def test_ingest_refuses_a_silent_default_rate(tmp_path):
    from brain_moe_pinn.data.ingest_c_elegans import convert_directory

    raw = tmp_path / "raw"
    _write_worm(raw, 1, n_frames=50, n_channels=2)
    with pytest.raises(ValueError, match="frame rate"):
        convert_directory(raw, tmp_path / "out", qc_autocorr=None)


# --------------------------------------------------------------------------- #
# A5/A6/A7: reference QC, overwrite, name-mapping filter
# --------------------------------------------------------------------------- #
def test_autocorr_qc_drops_noisy_channels(tmp_path):
    from brain_moe_pinn.data.ingest_c_elegans import convert_directory

    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    smooth = np.cumsum(rng.standard_normal(600))          # autocorrelated
    noise = rng.standard_normal(600)                      # white
    data = np.stack([smooth, noise], axis=1)
    raw.mkdir(parents=True, exist_ok=True)
    np.savetxt(raw / "1_ratio.csv", data, delimiter=",")
    (raw / "1_uniqNames.csv").write_text("A\nB\n")
    convert_directory(raw, tmp_path / "out", rate_hz=10.0, qc_autocorr=0.3,
                      stimulus_track=False)
    kept = np.load(tmp_path / "out" / "calcium" / "1.npy")
    ids = (tmp_path / "out" / "calcium_ids" / "1.txt").read_text().split()
    assert kept.shape[0] == 1 and ids == ["A"]


def test_apply_names_and_filter_known_keeps_mapped_channels(tmp_path):
    """Regression: the filter must test the *mapped* ids, not the raw tokens."""
    from brain_moe_pinn.data.ingest_c_elegans import convert_directory

    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.standard_normal((400, 3)), axis=0)
    raw.mkdir(parents=True, exist_ok=True)
    np.savetxt(raw / "2_ratio.csv", base, delimiter=",")
    (raw / "2_uniqNames.csv").write_text("1\n2\n3\n")     # positional ids
    out = tmp_path / "out"
    convert_directory(raw, out, rate_hz=5.0, qc_autocorr=None, stimulus_track=False,
                      apply_names=True, filter_known=True,
                      metadata_dir=_metadata_dir(tmp_path, ["A", "B", "C"]))
    assert np.load(out / "calcium" / "2.npy").shape == (3, 400)
    assert (out / "calcium_ids" / "2.txt").read_text().split() == ["A", "B", "C"]


def test_empty_channel_selection_is_an_error(tmp_path):
    from brain_moe_pinn.data.ingest_c_elegans import convert_directory

    raw = tmp_path / "raw"
    _write_worm(raw, 3, n_frames=60, n_channels=2)
    with pytest.raises(ValueError, match="kept 0"):
        convert_directory(raw, tmp_path / "out", rate_hz=5.0, qc_autocorr=None,
                          stimulus_track=False, filter_known=True,
                          metadata_dir=_metadata_dir(tmp_path, ["X"]))


def test_reingest_requires_overwrite(tmp_path):
    from brain_moe_pinn.data.ingest_c_elegans import convert_directory

    raw = tmp_path / "raw"
    _write_worm(raw, 4, n_frames=60, n_channels=2)
    out = tmp_path / "out"
    kwargs = dict(rate_hz=5.0, qc_autocorr=None, stimulus_track=False)
    convert_directory(raw, out, **kwargs)
    with pytest.raises(FileExistsError):
        convert_directory(raw, out, **kwargs)
    convert_directory(raw, out, overwrite=True, **kwargs)


def test_manifest_merge_adds_columns_without_crashing(tmp_path):
    path = tmp_path / "manifest.csv"
    _write_rows(path, [{"sample_id": "a", "subject": "1", "rate_hz": "5"}])
    _write_rows(path, [{"sample_id": "b", "subject": "2", "rate_hz": "4",
                        "animal": "9"}])
    rows = _read_manifest(path)
    assert {r["sample_id"] for r in rows} == {"a", "b"}
    assert rows[1]["animal"] == "9" and rows[0].get("animal", "") == ""


# --------------------------------------------------------------------------- #
# B3/B6: rate contract, per-sample dt, species tags
# --------------------------------------------------------------------------- #
def test_loader_rejects_rate_mismatch_with_the_profile(tmp_path):
    root = tmp_path / "ladder"
    emit_sample(root, "w1", {"calcium": np.random.randn(3, 120).astype("float32")},
                subject="1", rate_hz=4.0, dt_s=0.25)
    with pytest.raises(ValueError, match="sampling-rate mismatch"):
        SpeciesSignalDataset(root, ["calcium"], seq_len=16,
                             expected_rate_hz=10.0, random_windows=False)[0]
    # the manifest clock wins when the caller says the corpus is non-uniform
    item = SpeciesSignalDataset(root, ["calcium"], seq_len=16,
                                expected_rate_hz=10.0, strict_rate=False,
                                random_windows=False)[0]
    assert item["dt"] == pytest.approx(0.25)


def test_seconds_window_uses_the_manifest_rate(tmp_path):
    root = tmp_path / "ladder"
    emit_sample(root, "w1", {"calcium": np.random.randn(3, 400).astype("float32")},
                subject="1", rate_hz=4.0, dt_s=0.25)
    ds = SpeciesSignalDataset(root, ["calcium"], seq_seconds=10.0,
                              random_windows=False)
    assert ds[0]["calcium"].shape[-1] == 40       # 10 s at 4 Hz, not 10*10


def test_step_dt_changes_the_latent_advance(tmp_path):
    from brain_moe_pinn import BrainMoEPINNConfig

    cfg = BrainMoEPINNConfig(
        latent_dim=16, generic_observation_only=True,
        use_species_conditioning=False, use_kda_decoder=False,
        use_active_inference=False)
    model = cfg.to_model().eval()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        one = model.forward_modalities({"calcium": x}, num_steps=1, dt=1.0)
        ten = model.forward_modalities({"calcium": x}, num_steps=1, dt=10.0)
        per_sample = model.forward_modalities(
            {"calcium": x}, num_steps=1, dt=torch.tensor([1.0, 10.0]))
    # the physical duration scales the latent advance linearly: one latent
    # step advances dt seconds of model time
    step_one = one["z_next"] - one["z_global"]
    step_ten = ten["z_next"] - ten["z_global"]
    assert torch.allclose(step_ten, 10.0 * step_one, rtol=1e-4, atol=1e-6)
    # per-sample dt: each row advances by its own duration
    uniform = model.forward_modalities(
        {"calcium": x}, num_steps=1, dt=torch.tensor([1.0, 1.0]))
    step_uniform = uniform["z_next"] - uniform["z_global"]
    step_per = per_sample["z_next"] - per_sample["z_global"]
    assert torch.allclose(step_per[1], 10.0 * step_uniform[1],
                          rtol=1e-4, atol=1e-6)
    assert torch.allclose(step_per[0], step_uniform[0], rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="one value per sample"):
        model.forward_modalities({"calcium": x}, num_steps=1, dt=torch.ones(3))


# --------------------------------------------------------------------------- #
# C1/C4: reconstruction head and mask-aware adapters
# --------------------------------------------------------------------------- #
def test_reconstruction_head_gives_channels_distinct_waveforms():
    from brain_moe_pinn.core.observation_adapters import SignalReconstructionHead

    torch.manual_seed(0)
    head = SignalReconstructionHead(latent_dim=8, num_time_basis=16)
    out = head(torch.randn(2, 8), num_channels=4, target_time_len=32)
    demeaned = out[0] - out[0].mean(dim=-1, keepdim=True)
    # every channel must carry its own temporal shape, not a shared one
    spread = (demeaned - demeaned.mean(dim=0, keepdim=True)).abs().mean()
    assert float(spread.detach()) > 1e-3


def test_adapter_statistics_and_pooling_respect_the_mask():
    from brain_moe_pinn.core.observation_adapters import (
        ChannelSignalAdapter, GenericSignalAdapter,
    )

    torch.manual_seed(0)
    signal = torch.randn(1, 3, 32)
    mask = torch.ones(1, 3, 32, dtype=torch.bool)
    mask[0, 2] = False
    padded = signal.clone()
    padded[0, 2] = 0.0

    adapter = GenericSignalAdapter(latent_dim=4, kernel_size=4, stride=4)
    unmasked = adapter(padded)
    masked = adapter(padded, mask=mask)
    assert not torch.allclose(unmasked, masked)

    chan = ChannelSignalAdapter(latent_dim=4, kernel_size=4, stride=4)
    pooled_masked = chan(padded, mask=mask)
    all_valid = chan(padded, mask=torch.ones_like(mask))
    assert not torch.allclose(pooled_masked, all_valid)


# --------------------------------------------------------------------------- #
# C3/C5/C6: emission channel, baselines, parameter estimators
# --------------------------------------------------------------------------- #
def test_emission_model_saturates_and_is_learnable():
    from brain_moe_pinn.core.emission import SensorEmission

    emission = SensorEmission(max_channels=4, init_tau_s=1.5)
    signal = torch.randn(2, 4, 64)
    out = emission(signal, dt=0.25)
    assert out.shape == signal.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    summary = emission.parameter_summary(4)
    assert summary["tau_s_median"] == pytest.approx(1.5, abs=1e-3)
    loss = out.mean()
    loss.backward()
    assert emission.log_tau.grad is not None
    assert float(emission.log_tau.grad.abs().sum()) > 0


def test_recon_baseline_helper_matches_manual_correlation():
    from brain_moe_pinn.training.training_loop import masked_channel_correlation

    target = torch.randn(2, 3, 40)
    assert masked_channel_correlation(target, target) == pytest.approx(1.0, abs=1e-5)
    flipped = -target
    assert masked_channel_correlation(flipped, target) == pytest.approx(-1.0, abs=1e-5)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., :10] = True
    assert masked_channel_correlation(target, target, mask) == pytest.approx(1.0, abs=1e-5)


def test_tau_estimator_recovers_known_time_constant():
    from brain_moe_pinn.diagnostics.species_parameters import (
        coupling_split, estimate_tau,
    )

    rng = np.random.default_rng(0)
    dt, tau_true, n = 0.25, 2.0, 20000
    a = np.exp(-dt / tau_true)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + rng.standard_normal() * np.sqrt(1 - a ** 2)
    est = estimate_tau(x[None, :], dt_s=dt)
    assert est.summary()["median_tau_s"] == pytest.approx(tau_true, rel=0.05)

    symmetric = np.array([[0.0, 0.4, 0.0], [0.4, 0.0, 0.2], [0.0, 0.2, 0.0]])
    split = coupling_split(symmetric)
    assert split["symmetric_share"] == pytest.approx(1.0)
    directed = np.array([[0.0, 0.4, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert coupling_split(directed)["antisymmetric_share"] == pytest.approx(1.0)


def test_graph_directory_is_read(tmp_path):
    root = tmp_path / "ladder"
    emit_sample(root, "s1", {"calcium": np.random.randn(3, 40).astype("float32")},
                rate_hz=4.0, graph=np.eye(3, dtype="float32"))
    ds = SpeciesSignalDataset(root, ["calcium"], seq_len=8, load_graphs=True,
                              random_windows=False)
    assert ds[0]["graph"].shape == (3, 3)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_manifest(path: Path):
    import csv as _csv
    with open(path, newline="") as handle:
        return list(_csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# sensors: imaging methods x functional reporters
# --------------------------------------------------------------------------- #
def test_sensor_registry_resolves_imaging_and_reporter():
    from brain_moe_pinn.core.sensors import (
        IMAGING_METHODS, REPORTERS, resolve_sensor,
    )

    # every imaging method the ladder must support is registered
    for name in ("lsfm_spim", "lfm_xlfm", "rs_lsfm", "scape_3d_aod_2p",
                 "spinning_disk_4d", "widefield_2p", "electrical"):
        assert name in IMAGING_METHODS
    # reporter families: fast/slow/nuclear GCaMP, FRET cameleon, GEVI, pERK
    for name in ("gcamp6f", "gcamp6s", "gcamp7f", "jgcamp8f", "jgcamp8s",
                 "h2b_gcamp6s", "h2b_gcamp7f", "yc2.60", "positron2_kv",
                 "voltron", "arch", "perk"):
        assert name in REPORTERS

    worm = resolve_sensor("calcium", {
        "imaging": "spinning_disk_4d", "reporter": "yc2.60", "readout": "ratio"})
    assert worm.readout == "ratio" and worm.reporter.localization == "nuclear"
    # the worm kinetics are not established in the registry -> fit, do not import
    assert worm.calibration_required and worm.emission_tau_s is None

    fish = resolve_sensor("calcium", {
        "imaging": "lfm_xlfm", "reporter": "jgcamp8f"})
    assert fish.emission_tau_s == pytest.approx(0.067)
    assert fish.imaging.volume_integration_s is not None  # snapshot volume

    gevi = resolve_sensor("voltage", {
        "imaging": "lsfm_spim", "reporter": "positron2_kv"})
    assert gevi.readout == "voltage" and gevi.emission_tau_s == pytest.approx(5.1e-4)

    perk = resolve_sensor("calcium", {
        "imaging": "spinning_disk_4d", "reporter": "perk", "readout": "static"})
    assert perk.dynamics_valid is False


def test_sensor_registry_rejects_mismatched_combinations():
    from brain_moe_pinn.core.sensors import resolve_sensor

    # a voltage reporter cannot back the calcium modality
    with pytest.raises(ValueError, match="voltage reporter"):
        resolve_sensor("calcium", {"imaging": "lsfm_spim", "reporter": "voltron"})
    # a fixed-tissue marker cannot back a dynamic readout
    with pytest.raises(ValueError, match="history marker"):
        resolve_sensor("calcium", {"imaging": "spinning_disk_4d",
                                   "reporter": "perk", "readout": "dff"})
    # unknown names fail loudly with the known set
    with pytest.raises(ValueError, match="unknown reporter"):
        resolve_sensor("calcium", {"imaging": "lsfm_spim", "reporter": "gcamp9"})
    with pytest.raises(ValueError, match="unknown imaging method"):
        resolve_sensor("calcium", {"imaging": "mri", "reporter": "gcamp6f"})


def test_voltage_emission_skips_the_hill_term():
    import torch as _torch
    from brain_moe_pinn.core.emission import SensorEmission
    from brain_moe_pinn.core.sensors import resolve_sensor

    spec = resolve_sensor("voltage", {"imaging": "lsfm_spim",
                                      "reporter": "positron2_kv"})
    emission = SensorEmission(max_channels=3, spec=spec)
    out = emission(_torch.randn(1, 3, 32) * 5.0, dt=0.002)
    # linear readout: the output is not squashed into [0, 1]
    assert float(out.abs().max()) > 1.0
    summary = emission.parameter_summary(3)
    assert summary["readout"] == "voltage" and "hill_h_median" not in summary


def test_profile_with_a_static_marker_cannot_request_next_step_targets():
    from brain_moe_pinn.config import ExperimentConfig

    with pytest.raises(ValueError, match="history marker"):
        ExperimentConfig.from_dict({
            "species": "mouse",
            "features": {"use_sensor_emission": False},
            "data": {
                "modalities": ["calcium"],
                "paired_next_step_targets": True,
                "sensors": {"calcium": {"imaging": "spinning_disk_4d",
                                        "reporter": "perk",
                                        "readout": "static"}},
            },
        })


def test_species_profiles_keep_separate_time_and_observation_contracts():
    """Each ladder stage trains its own model: rates and sensors must not blend."""
    from brain_moe_pinn.config import ExperimentConfig

    root = Path(__file__).resolve().parents[1]
    worm = ExperimentConfig.from_file(root / "configs" / "species" / "c_elegans.json")
    fish = ExperimentConfig.from_file(root / "configs" / "species" / "zebrafish.json")

    # sampling contracts differ and are manifest-sourced on both stages
    assert worm.data.sample_rate_hz_source == "manifest"
    assert fish.data.sample_rate_hz_source == "manifest"
    assert worm.data.sample_rate_hz != fish.data.sample_rate_hz

    worm_sensor = worm.sensor_specs()["calcium"]
    fish_sensor = fish.sensor_specs()["calcium"]
    assert (worm_sensor.reporter.name, worm_sensor.readout) == ("yc2.60", "ratio")
    assert (fish_sensor.reporter.name, fish_sensor.readout) == ("h2b_gcamp7f", "dff")
    assert worm_sensor.imaging.name == "spinning_disk_4d"
    assert fish_sensor.imaging.name == "lsfm_spim"
    # the two acquisition schemes have different frame-integration and smear
    assert (worm_sensor.imaging.volume_integration_s
            != fish_sensor.imaging.volume_integration_s)
    # a zebrafish voltage stage is a different observation channel again
    assert fish.sensor_specs()["voltage"].readout == "voltage"
