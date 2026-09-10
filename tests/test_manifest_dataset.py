"""P0-P2 loader features: manifest rows, subject splits, masks, union
alignment, region aggregation, seconds windows, trials, readers.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from brain_moe_pinn.data.readers import emit_sample, summarize_ladder
from brain_moe_pinn.data.species_dataset import (
    SpeciesSignalDataset, build_species_dataloaders, split_rows,
)
from brain_moe_pinn.training.losses import TotalLoss
from brain_moe_pinn.training.training_phases import LossWeights


def _write_npy(directory: Path, stem: str, array) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{stem}.npy", array)


def _write_ids(directory: Path, stem: str, ids) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.txt").write_text("\n".join(ids) + "\n")


def _manifest(root: Path, rows) -> None:
    import csv
    with open(root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_subject_grouped_splits_are_disjoint():
    rows = [
        {"sample_id": f"{s}{i}", "subject": s, "session": "1",
         "origin": "o1", "condition": "c1"}
        for s in ("a", "b", "c", "d") for i in range(3)
    ]
    train, val, test = split_rows(rows, by="subject", seed=1,
                                  val_frac=0.25, test_frac=0.25)
    train_sub = {r["subject"] for r in train}
    val_sub = {r["subject"] for r in val}
    test_sub = {r["subject"] for r in test}
    assert train_sub and val_sub and test_sub
    assert not (train_sub & val_sub | train_sub & test_sub | val_sub & test_sub)
    # determinism
    assert split_rows(rows, by="subject", seed=1, val_frac=0.25, test_frac=0.25)[0] == train


def test_manifest_federation_masks_and_splits(tmp_path):
    for s in ("a1", "a2", "b1", "b2"):
        _write_npy(tmp_path / "calcium", s, np.random.randn(4, 60))
        _write_ids(tmp_path / "calcium_ids", s, ["n1", "n2", "n3", "n4"])
        mask = np.ones((4, 60), dtype=bool)
        mask[0] = False  # first channel invalid
        _write_npy(tmp_path / "calcium_mask", s, mask)
    # b1/b2 also carry voltage (federation example: not all subjects)
    _write_npy(tmp_path / "voltage", "b1", np.random.randn(2, 60))
    _write_npy(tmp_path / "voltage", "b2", np.random.randn(2, 60))
    rows = [
        {"sample_id": s, "subject": s[0], "session": "1",
         "origin": "o1", "condition": "x", "rate_hz": "10"}
        for s in ("a1", "a2", "b1", "b2")
    ]
    _manifest(tmp_path, rows)

    ds = SpeciesSignalDataset(
        tmp_path, ["calcium", "voltage"], seq_len=32,
        load_ids=True, return_masks=True)
    assert len(ds) == 4
    item = ds[0]
    assert "calcium" in item and "calcium_mask" in item
    assert not item["calcium_mask"][0].any()  # invalid channel honored
    # voltage absent for subject a -> dropped per row (federation)
    assert "voltage" not in item

    train_loader, val_loader = build_species_dataloaders(
        tmp_path, ["calcium", "voltage"], batch_size=2, seq_len=32,
        split_by="subject", seed=0, val_frac=0.5, load_ids=True)
    train_sub = {b["subject"][0] for b in train_loader}
    val_sub = {b["subject"][0] for b in val_loader}
    assert not (train_sub & val_sub)
    # batches drop partial-key modalities only when heterogeneous per batch
    for batch in train_loader:
        if "voltage" in batch:
            assert batch["voltage"].shape[1] == 2


def test_union_alignment_and_region_aggregation(tmp_path):
    # two samples with overlapping channel sets
    _write_npy(tmp_path / "calcium", "s1", np.random.randn(2, 40))
    _write_ids(tmp_path / "calcium_ids", "s1", ["n1", "n2"])
    _write_npy(tmp_path / "calcium", "s2", np.random.randn(2, 40))
    _write_ids(tmp_path / "calcium_ids", "s2", ["n2", "n3"])
    rows = [{"sample_id": s, "subject": s, "session": "1"} for s in ("s1", "s2")]
    _manifest(tmp_path, rows)

    ds = SpeciesSignalDataset(
        tmp_path, ["calcium"], seq_len=16, manifest=None, rows=rows,
        align_channels=True, load_ids=True, return_masks=True)
    item0, item1 = ds[0], ds[1]
    assert item0["calcium"].shape == (3, 16)  # union n1,n2,n3
    assert not item0["calcium_mask"][2].any()  # s1 lacks n3
    assert not item1["calcium_mask"][0].any()  # s2 lacks n1

    # region aggregation: n1,n2 -> R1; n3 -> R2
    (tmp_path / "regions.csv").write_text(
        "channel_id,region\nn1,R1\nn2,R1\nn3,R2\n")
    ds_r = SpeciesSignalDataset(
        tmp_path, ["calcium"], seq_len=16, manifest=None, rows=rows,
        region_map=tmp_path / "regions.csv", load_ids=True,
        return_masks=False)
    item_r = ds_r[0]
    assert item_r["calcium"].shape == (2, 16)
    assert item_r["calcium_ids"] == ["R1", "R2"]


def test_seconds_windows_pad_time_with_mask(tmp_path):
    _write_npy(tmp_path / "calcium", "f1", np.random.randn(3, 20))  # 10 Hz
    _write_npy(tmp_path / "calcium", "f2", np.random.randn(3, 80))  # 20 Hz
    rows = [
        {"sample_id": "f1", "subject": "a", "session": "1",
         "origin": "o", "condition": "x", "rate_hz": "10"},
        {"sample_id": "f2", "subject": "b", "session": "1",
         "origin": "o", "condition": "x", "rate_hz": "20"},
    ]
    _manifest(tmp_path, rows)
    train_loader, _ = build_species_dataloaders(
        tmp_path, ["calcium"], batch_size=2, seq_len=None,
        seq_seconds=2.0, split_by="none", val_frac=0.0, shuffle=False)
    batch = next(iter(train_loader))
    # f1 -> 20 frames, f2 -> 40 frames; collator pads to 40
    assert batch["calcium"].shape == (2, 3, 40)
    assert not batch["calcium_mask"][0, :, 20:].any()
    assert batch["calcium_mask"][1].all()


def test_trial_windows_respected(tmp_path):
    _write_npy(tmp_path / "calcium", "s1", np.random.randn(4, 100))
    trials = tmp_path / "calcium_trials"
    trials.mkdir(parents=True, exist_ok=True)
    (trials / "s1.json").write_text(json.dumps([[0, 50], [80, 100]]))
    rows = [{"sample_id": "s1", "subject": "s1", "session": "1",
             "rate_hz": "10"}]
    _manifest(tmp_path, rows)
    ds = SpeciesSignalDataset(tmp_path, ["calcium"], seq_len=20, use_trials=True)
    allowed = set(range(0, 31)) | {80}
    starts = []
    for _ in range(20):
        window = ds[0]["calcium"]
        # recover start: compare to the source array prefix
        full = np.load(tmp_path / "calcium" / "s1.npy")
        match = [t for t in range(0, 81)
                 if np.allclose(full[:, t:t + 20], window.numpy())]
        assert match
        assert match[0] in allowed
        starts.append(match[0])
    assert set(starts) <= allowed


def test_masked_recon_total_loss(tmp_path):
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4, 16, dtype=torch.bool)
    mask[0, 1] = False
    weights = LossWeights(recon_extra={"calcium": 1.0}, sigreg=0.0)
    tl = TotalLoss(weights)
    _, metrics = tl({"calcium_recon": pred},
                    {"calcium": target, "calcium_mask": mask})
    manual = ((pred - target) ** 2)[mask].mean()
    assert torch.allclose(torch.tensor(metrics["recon_calcium"]), manual)


def test_readers_emit_and_summarize(tmp_path):
    from brain_moe_pinn.data.readers import read_manifest
    row = emit_sample(
        tmp_path, "w1", {"calcium": np.random.randn(4, 50)},
        subject="w1", origin="test", condition="salt", rate_hz=10.0,
        ids={"calcium": ["n1", "n2", "n3", "n4"]},
        masks={"calcium": np.ones((4, 50), dtype=bool)},
        trials={"calcium": [(0, 50)]},
        graph=np.eye(4))
    assert row["sample_id"] == "w1"
    assert (tmp_path / "calcium" / "w1.npy").exists()
    assert (tmp_path / "calcium_ids" / "w1.txt").read_text().splitlines() == [
        "n1", "n2", "n3", "n4"]
    assert (tmp_path / "graphs" / "w1.npy").exists()
    # second emit merges into manifest
    emit_sample(tmp_path, "w2", {"calcium": np.random.randn(4, 50)},
                subject="w2")
    assert len(read_manifest(tmp_path / "manifest.csv")) == 2
    report = summarize_ladder(tmp_path)
    assert report["calcium"]["samples"] == 2
