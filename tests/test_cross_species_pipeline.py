"""Regression tests for cross-species and human corpus pipelines."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from brain_moe_pinn.config import ExperimentConfig, SUPPORTED_MODALITIES
from brain_moe_pinn.data.corpus_pipeline import (
    encode_optogenetic_control,
    ingest_bids_fmri_session,
    ingest_bids_session,
    ingest_source_manifest,
    validate_ladder,
)
from brain_moe_pinn.data.species_dataset import SpeciesSignalDataset, build_species_dataloaders


def test_zebrafish_source_manifest_emits_multirate_ladder(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    rng = np.random.default_rng(4)
    calcium = rng.normal(size=(40, 3)).astype("float32")
    calcium[5, 1] = np.nan
    stimulus = rng.normal(size=(40, 2)).astype("float32")
    np.savez(source / "dryad_calcium.npz", traces=calcium)
    np.save(source / "dryad_stimulus.npy", stimulus)
    (source / "calcium_ids.txt").write_text("cell_a\ncell_b\ncell_c\n")
    manifest = {
        "species": "zebrafish",
        "samples": [{
            "sample_id": "dryad__larva01",
            "subject": "larva01",
            "session": "visual_habituation_01",
            "origin": "Dryad:10.5061/dryad.jdfn2z3fc",
            "condition": "dark_flash",
            "modalities": {
                "calcium": {
                    "path": "dryad_calcium.npz",
                    "key": "traces",
                    "orientation": "tc",
                    "rate_hz": 2.0,
                    "ids": ["cell_a", "cell_b", "cell_c"]
                },
                "stimulus": {
                    "path": "dryad_stimulus.npy",
                    "orientation": "tc",
                    "rate_hz": 5.0
                }
            }
        }]
    }
    source_manifest = source / "manifest.json"
    source_manifest.write_text(json.dumps(manifest))
    out = tmp_path / "zebrafish_ladder"

    ingest_source_manifest(source_manifest, out)
    report = validate_ladder(out, modalities=("calcium", "stimulus"))
    assert report["samples"] == 1
    assert report["modalities"]["calcium"]["channels"] == [3]
    assert np.load(out / "calcium" / "dryad__larva01.npy").shape == (3, 40)
    assert np.load(out / "calcium_mask" / "dryad__larva01.npy")[1, 5] == 0

    dataset = SpeciesSignalDataset(
        out,
        ["calcium", "stimulus"],
        seq_seconds=2.0,
        roles={"calcium": "signal", "stimulus": "control"},
        species="zebrafish",
        return_next_step_targets=True,
        random_windows=False,
    )
    item = dataset[0]
    assert item["calcium"].shape == (3, 4)
    assert item["stimulus"].shape == (2, 10)
    assert item["calcium_next"].shape == (3, 4)
    assert item["stimulus_next"].shape == (2, 10)
    assert item["dt"] == pytest.approx(0.5)
    assert item["species"] == "zebrafish"

    train, val = build_species_dataloaders(
        out,
        ["calcium", "stimulus"],
        batch_size=1,
        seq_seconds=2.0,
        split_by="none",
        val_frac=0.0,
        shuffle=False,
        roles={"calcium": "signal", "stimulus": "control"},
        species="zebrafish",
        return_next_step_targets=True,
    )
    batch = next(iter(train))
    assert batch["calcium"].shape == (1, 3, 4)
    assert batch["stimulus"].shape == (1, 2, 10)
    assert len(val.dataset) == 0

def test_optogenetic_control_gates_target_code_and_manifest_metadata(tmp_path):
    waveform = np.array([[0, 1, 0, 2, 0, 0, 0, 0]], dtype="float32")
    encoded, feature_ids = encode_optogenetic_control(
        waveform, ["AVAL"], ["AVAL", "AVAR"])
    assert encoded.shape == (3, 8)
    assert feature_ids[-2:] == ["target:AVAL", "target:AVAR"]
    assert np.array_equal(encoded[1], waveform[0])
    assert np.count_nonzero(encoded[2]) == 0
    assert np.count_nonzero(encoded[:, 0]) == 0

    source = tmp_path / "opto_source"
    source.mkdir()
    np.save(source / "waveform.npy", waveform)
    (source / "target_vocab.txt").write_text("AVAL\nAVAR\n")
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "species": "c_elegans",
        "samples": [{
            "sample_id": "randi__AVAL",
            "subject": "worm01",
            "session": "opto01",
            "origin": "Randi et al.",
            "condition": "AVAL_stimulation",
            "modalities": {
                "opto": {
                    "path": "waveform.npy",
                    "orientation": "ct",
                    "rate_hz": 4.0,
                    "target_id": "AVAL",
                    "target_vocab": "target_vocab.txt",
                }
            },
        }],
    }))
    out = tmp_path / "opto_ladder"
    ingest_source_manifest(manifest, out)
    report = validate_ladder(out, modalities=("opto",))
    assert report["modalities"]["opto"]["channels"] == [3]
    assert np.load(out / "opto" / "randi__AVAL.npy").shape == (3, 8)
    manifest_text = (out / "manifest.csv").read_text()
    assert ",optogenetic," in manifest_text
    assert ",AVAL," in manifest_text
    assert "target_vocab_size" in manifest_text
    assert "target:AVAL|target:AVAR" in manifest_text



def test_global_zscore_uses_training_subjects_only(tmp_path):
    root = tmp_path / "ladder"
    (root / "calcium").mkdir(parents=True)
    np.save(root / "calcium" / "train.npy",
            np.arange(4, dtype="float32").reshape(1, 4))
    np.save(root / "calcium" / "validation.npy",
            (100 + np.arange(4, dtype="float32")).reshape(1, 4))
    (root / "manifest.csv").write_text(
        "sample_id,subject,session,rate_hz,dt_s\n"
        "train,subject_train,1,1,1\n"
        "validation,subject_validation,1,1,1\n")

    train, val = build_species_dataloaders(
        root,
        ["calcium"],
        batch_size=1,
        seq_len=2,
        split_by="subject",
        val_frac=0.5,
        shuffle=False,
        roles={"calcium": "signal"},
        normalization="global_zscore",
    )
    train_row = train.dataset.rows[0]
    train_values = np.load(
        root / "calcium" / f"{train_row['sample_id']}.npy")
    expected_mean = float(train_values.mean())
    expected_std = float(train_values.std())
    stats = train.dataset.normalization_stats["calcium"]
    assert stats["mean"] == pytest.approx(expected_mean)
    assert stats["std"] == pytest.approx(expected_std)
    assert val.dataset.normalization_stats == train.dataset.normalization_stats
    val_batch = next(iter(val))["calcium"]
    assert abs(float(val_batch.mean())) > 50.0


def test_ambiguous_square_source_requires_orientation(tmp_path):
    source = tmp_path / "square.npy"
    np.save(source, np.ones((4, 4), dtype="float32"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "species": "mouse",
        "samples": [{
            "sample_id": "square",
            "modalities": {"calcium": {"path": source.name}}
        }]
    }))
    with pytest.raises(ValueError, match="ambiguous orientation"):
        ingest_source_manifest(manifest, tmp_path / "out")


def test_mouse_bids_fmri_ingest_parcels_and_reads_tr(tmp_path):
    nib = pytest.importorskip("nibabel")
    bids = tmp_path / "ds004402"
    bold_dir = bids / "sub-01" / "func"
    bold_dir.mkdir(parents=True)
    data = np.arange(2 * 2 * 2 * 20, dtype="float32").reshape(2, 2, 2, 20)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_zooms((1.0, 1.0, 1.0, 2.0))
    bold = bold_dir / "sub-01_task-odor_bold.nii.gz"
    nib.save(image, bold)
    (bold_dir / "sub-01_task-odor_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0}))
    atlas = tmp_path / "atlas.nii.gz"
    labels = np.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]], dtype="float32")
    nib.save(nib.Nifti1Image(labels, np.eye(4)), atlas)
    out = tmp_path / "mouse_ladder"

    ingest_bids_fmri_session(
        bids, out, subject="01", task="odor", atlas=atlas,
        origin="OpenNeuro:ds004402")
    report = validate_ladder(out, modalities=("fmri",))
    assert report["modalities"]["fmri"]["channels"] == [2]
    assert np.load(out / "fmri" / "openneuro__sub-01__task-odor.npy").shape == (2, 20)
    row = (out / "manifest.csv").read_text()
    assert "fmri_rate_hz" in row and "0.5" in row
    assert (out / "fmri_ids" / "openneuro__sub-01__task-odor.txt").read_text().splitlines() == [
        "region_0001", "region_0002"
    ]

def test_human_multimodal_source_manifest_preserves_modality_clocks(tmp_path):
    source = tmp_path / "human_source"
    source.mkdir()
    rng = np.random.default_rng(9)
    eeg = rng.normal(size=(64, 32)).astype("float32")
    eeg[0, 3] = np.nan
    meg = rng.normal(size=(306, 32)).astype("float32")
    fmri = rng.normal(size=(400, 8)).astype("float32")
    np.save(source / "eeg.npy", eeg)
    np.save(source / "meg.npy", meg)
    np.save(source / "fmri.npy", fmri)
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "species": "human",
        "samples": [{
            "sample_id": "openneuro__sub-01__task-rest",
            "subject": "sub-01",
            "session": "ses-01",
            "origin": "OpenNeuro:ds006040",
            "condition": "rest",
            "cross_modal_label": 1,
            "modalities": {
                "eeg": {
                    "format": "array",
                    "path": "eeg.npy",
                    "orientation": "ct",
                    "rate_hz": 256,
                    "ids": [f"EEG-{i:03d}" for i in range(64)],
                },
                "meg": {
                    "format": "array",
                    "path": "meg.npy",
                    "orientation": "ct",
                    "rate_hz": 256,
                    "ids": [f"MEG-{i:03d}" for i in range(306)],
                },
                "fmri": {
                    "format": "array",
                    "path": "fmri.npy",
                    "orientation": "ct",
                    "rate_hz": 0.5,
                    "ids": [f"region_{i:04d}" for i in range(400)],
                },
            },
        }],
    }))
    out = tmp_path / "human_ladder"

    ingest_source_manifest(manifest, out)
    report = validate_ladder(out, modalities=("eeg", "meg", "fmri"))
    assert report["samples"] == 1
    assert report["modalities"]["eeg"]["channels"] == [64]
    assert report["modalities"]["meg"]["channels"] == [306]
    assert report["modalities"]["fmri"]["channels"] == [400]
    assert not np.load(
        out / "eeg_mask" / "openneuro__sub-01__task-rest.npy")[0, 3]
    manifest_text = (out / "manifest.csv").read_text()
    assert "eeg_rate_hz" in manifest_text
    assert "meg_rate_hz" in manifest_text
    assert "fmri_rate_hz" in manifest_text
    assert manifest_text.rstrip().endswith(",1")


def test_mne_source_fails_with_actionable_optional_dependency(tmp_path):
    import importlib.util
    if importlib.util.find_spec("mne") is not None:
        pytest.skip("MNE is installed; parser-specific coverage is environment-dependent")
    source = tmp_path / "mne_source"
    source.mkdir()
    recording = source / "sub-01_task-rest_eeg.edf"
    recording.write_text("not an EDF file")
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "species": "human",
        "samples": [{
            "sample_id": "mne",
            "modalities": {
                "eeg": {
                    "format": "mne",
                    "path": recording.name,
                    "modality": "eeg",
                },
            },
        }],
    }))
    with pytest.raises(ImportError, match="mne-python"):
        ingest_source_manifest(manifest, tmp_path / "out")


def test_mne_fif_source_selects_eeg_channels_and_resamples(tmp_path):
    mne = pytest.importorskip("mne")
    info = mne.create_info(
        ["EEG001", "EOG001", "EEG002"],
        sfreq=100.0,
        ch_types=["eeg", "eog", "eeg"],
    )
    raw = mne.io.RawArray(np.arange(150, dtype="float64").reshape(3, 50), info)
    recording = tmp_path / "sub-01_task-rest_eeg.fif"
    raw.save(recording, overwrite=True, verbose="ERROR")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "species": "human",
        "samples": [{
            "sample_id": "fif",
            "modalities": {
                "eeg": {
                    "format": "fif",
                    "path": recording.name,
                    "modality": "eeg",
                    "channel_types": ["eeg"],
                    "target_rate_hz": 50,
                },
            },
        }],
    }))

    out = tmp_path / "out"
    ingest_source_manifest(manifest, out)
    assert np.load(out / "eeg" / "fif.npy").shape == (2, 25)
    assert (out / "eeg_ids" / "fif.txt").read_text().splitlines() == [
        "EEG001", "EEG002"
    ]
    row = (out / "manifest.csv").read_text()
    assert "eeg_rate_hz" in row and "50" in row


def test_bids_session_joins_modalities_without_collapsing_clocks(tmp_path):
    mne = pytest.importorskip("mne")
    nib = pytest.importorskip("nibabel")
    bids = tmp_path / "bids"
    eeg_dir = bids / "sub-01" / "eeg"
    func_dir = bids / "sub-01" / "func"
    eeg_dir.mkdir(parents=True)
    func_dir.mkdir(parents=True)

    info = mne.create_info(["EEG001"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.ones((1, 50)), info)
    raw.save(eeg_dir / "sub-01_task-rest_eeg.fif",
             overwrite=True, verbose="ERROR")

    bold = func_dir / "sub-01_task-rest_bold.nii.gz"
    bold_data = np.arange(2 * 2 * 2 * 4, dtype="float32").reshape(2, 2, 2, 4)
    image = nib.Nifti1Image(bold_data, np.eye(4))
    image.header.set_zooms((1.0, 1.0, 1.0, 2.0))
    nib.save(image, bold)
    (func_dir / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0}))
    atlas = tmp_path / "atlas.nii.gz"
    labels = np.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]],
                      dtype="float32")
    nib.save(nib.Nifti1Image(labels, np.eye(4)), atlas)

    out = tmp_path / "ladder"
    ingest_bids_session(
        bids, out, subject="01", task="rest",
        modalities=("eeg", "fmri"), eeg_target_rate_hz=50,
        atlas=atlas, origin="OpenNeuro:ds006040",
        cross_modal_label=1,
    )
    report = validate_ladder(out, modalities=("eeg", "fmri"))
    assert report["samples"] == 1
    sample_id = "openneuro__sub-01__task-rest"
    assert np.load(out / "eeg" / f"{sample_id}.npy").shape == (1, 25)
    assert np.load(out / "fmri" / f"{sample_id}.npy").shape == (2, 4)
    with (out / "manifest.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["rate_hz"] == "50.0"
    assert row["eeg_rate_hz"] == "50"
    assert row["fmri_rate_hz"] == "0.5"
    assert row["cross_modal_label"] == "1"


def test_human_profile_uses_manifest_clocks():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/data/human_bids.json").open() as handle:
        profile = json.load(handle)
    assert profile["kind"] == "species"
    assert profile["modalities"] == ["eeg", "meg", "fmri"]
    assert profile["split_by"] == "subject"
    assert profile["return_masks"] is True
    human = ExperimentConfig.from_file(root / "configs/species/human.json")
    assert human.data.sample_rate_hz_source == "manifest"



def test_species_configs_declare_federated_rates_and_roles():
    root = Path(__file__).resolve().parents[1]
    fish = ExperimentConfig.from_file(root / "configs/species/zebrafish.json")
    mouse = ExperimentConfig.from_file(root / "configs/species/mouse.json")
    assert "stimulus" in SUPPORTED_MODALITIES
    assert "opto" in SUPPORTED_MODALITIES
    assert fish.data.sample_rate_hz_source == "manifest"
    assert fish.data.roles["stimulus"] == "control"
    assert fish.data.control_specs["opto"]["no_op_when_off"] is True
    assert mouse.data.sample_rate_hz_source == "manifest"
    assert mouse.data.control_specs["opto"]["target_encoding"] == "gated_one_hot"
    assert mouse.sensor_specs()["fmri"].readout == "bold"
    assert mouse.sensor_specs()["voltage"].imaging.name == "electrical"
    for name in ("zebrafish_crossrepo.json", "mouse_crossrepo.json"):
        with (root / "configs/data" / name).open() as handle:
            profile = json.load(handle)
        assert profile["kind"] == "species"
        assert profile["split_by"] == "subject"
        assert profile["return_next_step_targets"] is True
