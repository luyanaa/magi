"""Merge the three C. elegans canonical ladders into one federated root.

The source adapters remain separate. This boundary harmonizes their sample
namespace and presents one ``stimulus`` control modality to the generic
training path:

* Toyoshima/gKDR-GMM salt records keep their calcium channels and expose the
  scalar salt waveform in control feature 0.
* Randi PumpProbe records keep their calcium channels and expose target-gated
  opto features (feature 0 is the waveform; features 1..281 are
  recording-local target positions).
* HuggingFace worms contribute calcium-only rows; an absent control means no
  intervention, not a fabricated zero-valued modality on disk.

By default the output is a small manifest-federated root. Its manifest points
at the three existing canonical roots, avoiding a second multi-hundred-MB
copy on synced filesystems. Use ``--materialize`` when a self-contained copy
is required for transfer; preserve the relative source paths or dereference
those references while packaging a remote job.

The output uses the existing ``stimulus`` control modality. The training path
pads narrower source controls to the configured ``control_width`` before
reduction. Calcium identities remain source-local and the unified profile must
keep ``align_channels=false``. Subject IDs are source-qualified to prevent
accidental cross-source grouping.

Usage::

    python -m brain_moe_pinn.data.merge_c_elegans \\
        --toyoshima ../data_ladder/c_elegans_salt \\
        --randi ../data_ladder/c_elegans_randi \\
        --huggingface ../data_ladder/c_elegans_hf \\
        --output ../data_ladder/c_elegans_unified
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from .readers import _write_rows
from .species_dataset import read_manifest

_REQUIRED_SOURCES = ("toyoshima", "randi", "huggingface")
_SOURCE_PREFIX = {
    "toyoshima": "toyoshima_salt",
    "randi": "randi_pumpprobe",
    "huggingface": "hf_activity",
}
_SOURCE_ORIGIN = {
    "toyoshima": "Toyoshima et al. (2024) gKDR-GMM salt",
    "randi": "Randi et al. (2023) PumpProbe",
    "huggingface": "Simeon et al. (2024) HuggingFace celegans_neural_data",
}
_COMPANION_DIRS = (
    "calcium_ids",
    "calcium_mask",
    "calcium_trials",
    "stimulus_trials",
    "opto_events",
    "opto_trials",
    "randi_labels",
)
def _relative_reference(source: Path, output_root: Path) -> str:
    """Return a lexical manifest path that works when roots stay co-located."""
    return os.path.relpath(os.fspath(source), os.fspath(output_root))


def _materialize_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _control_source(
    source_name: str,
    source_root: Path,
    source_id: str,
    frames: int,
    control_width: int,
) -> tuple[Optional[Path], str]:
    """Validate and return the source control path plus its semantic kind."""
    if source_name == "toyoshima":
        path = source_root / "stimulus" / f"{source_id}.npy"
        kind = "toyoshima_salt_scalar"
        expected = (1, frames)
    elif source_name == "randi":
        path = source_root / "opto" / f"{source_id}.npy"
        kind = "randi_target_gated_opto"
        expected = (control_width, frames)
    elif source_name == "huggingface":
        return None, "none"
    else:
        raise ValueError(f"unknown C. elegans source {source_name!r}")

    if not path.exists():
        raise FileNotFoundError(
            f"{source_name} sample {source_id!r} has no control: {path}")
    control = np.load(path, mmap_mode="r")
    if tuple(control.shape) != expected:
        raise ValueError(
            f"control {path} must be {expected}, got {tuple(control.shape)}")
    return path, kind


def _materialize_control(
    source_path: Path,
    source_name: str,
    frames: int,
    control_width: int,
    destination: Path,
) -> None:
    """Write a fixed-width control array for optional self-contained output."""
    source = np.asarray(np.load(source_path), dtype=np.float32)
    if source_name == "toyoshima":
        control = np.zeros((control_width, frames), dtype=np.float32)
        control[0] = source[0]
    else:
        control = source
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, control)


def _add_companion_references(
    source_root: Path,
    output_root: Path,
    source_id: str,
    sample_id: str,
    row: Dict[str, str],
    *,
    materialize: bool,
) -> None:
    """Preserve IDs, masks, trial metadata, labels, and event JSON."""
    for directory in _COMPANION_DIRS:
        suffix = (
            ".json" if directory.endswith(("events", "trials"))
            else ".txt" if directory.endswith(("ids", "labels"))
            else ".npy"
        )
        source = source_root / directory / f"{source_id}{suffix}"
        if not source.exists():
            continue
        key = f"{directory}_file"
        if materialize:
            destination = output_root / directory / f"{sample_id}{suffix}"
            _materialize_copy(source, destination)
        else:
            row[key] = _relative_reference(source, output_root)


def merge_c_elegans_ladders(
    output_dir: Path,
    sources: Mapping[str, Path],
    *,
    control_width: int = 282,
    overwrite: bool = False,
    materialize: bool = False,
) -> Path:
    """Merge Toyoshima, Randi, and HuggingFace ladders.

    ``sources`` must contain the keys ``toyoshima``, ``randi``, and
    ``huggingface``. The default output stores source-relative file references
    in one manifest. ``materialize=True`` copies arrays and companions into a
    self-contained output root and pads the Toyoshima scalar control to the
    common width.
    """
    if control_width <= 0:
        raise ValueError("control_width must be positive")
    source_paths = {name: Path(sources[name]) for name in _REQUIRED_SOURCES}
    for name, root in source_paths.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{name} ladder root not found: {root}")
        if not (root / "manifest.csv").exists():
            raise FileNotFoundError(
                f"{name} ladder has no manifest.csv: {root}")

    output_root = Path(output_dir)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} exists; pass overwrite=True to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    counts = {name: 0 for name in _REQUIRED_SOURCES}
    controls = {name: 0 for name in _REQUIRED_SOURCES}
    for source_name in _REQUIRED_SOURCES:
        source_root = source_paths[source_name]
        prefix = _SOURCE_PREFIX[source_name]
        rows = read_manifest(source_root / "manifest.csv")
        if not rows:
            raise ValueError(f"{source_name} ladder manifest is empty")
        for source_row in rows:
            source_id = source_row.get("sample_id", "")
            if not source_id:
                raise ValueError(f"{source_name} manifest has an empty sample_id")
            sample_id = f"{prefix}__{source_id}"
            calcium = source_root / "calcium" / f"{source_id}.npy"
            if not calcium.exists():
                raise FileNotFoundError(
                    f"{source_name} sample {source_id!r} has no calcium: {calcium}")
            calcium_array = np.load(calcium, mmap_mode="r")
            if calcium_array.ndim != 2 or calcium_array.shape[0] < 1:
                raise ValueError(
                    f"{calcium} must be a non-empty (C,T) array")
            frames = int(calcium_array.shape[1])
            control_path, control_kind = _control_source(
                source_name, source_root, source_id, frames, control_width)

            row: Dict[str, str] = {
                "sample_id": sample_id,
                "subject": f"{prefix}::{source_row.get('subject') or source_id}",
                "session": source_row.get("session") or "1",
                "origin": source_row.get("origin") or _SOURCE_ORIGIN[source_name],
                "condition": source_row.get("condition") or "",
                "rate_hz": source_row.get("rate_hz") or "",
                "dt_s": source_row.get("dt_s") or "",
                "source_dataset": source_name,
                "source_sample_id": source_id,
                "source_origin": source_row.get("origin") or _SOURCE_ORIGIN[source_name],
                "control_kind": control_kind,
                "control_width": str(control_width),
                "control_present": "1" if control_path is not None else "0",
                "calcium_frame_count": str(frames),
                "calcium_channel_count": str(int(calcium_array.shape[0])),
            }
            if materialize:
                _materialize_copy(
                    calcium, output_root / "calcium" / f"{sample_id}.npy")
            else:
                row["calcium_file"] = _relative_reference(calcium, output_root)
            _add_companion_references(
                source_root, output_root, source_id, sample_id, row,
                materialize=materialize)

            if control_path is not None:
                controls[source_name] += 1
                if materialize:
                    _materialize_control(
                        control_path, source_name, frames, control_width,
                        output_root / "stimulus" / f"{sample_id}.npy")
                else:
                    # The loader and training partition pad the Toyoshima
                    # scalar (1,T) to control_width; Randi is already wide.
                    row["stimulus_file"] = _relative_reference(
                        control_path, output_root)

            for key, value in source_row.items():
                if key in {
                    "sample_id", "subject", "session", "origin", "condition",
                    "rate_hz", "dt_s",
                }:
                    continue
                if value not in (None, ""):
                    row[f"source_{key}"] = str(value)
            row["source_dataset"] = source_name
            row["source_sample_id"] = source_id
            row["source_origin"] = (
                source_row.get("origin") or _SOURCE_ORIGIN[source_name])
            manifest_rows.append(row)
            counts[source_name] += 1

    _write_rows(output_root / "manifest.csv", manifest_rows)
    vocab = ["scalar_drive"] + [
        f"randi_recording_local_target_{i:03d}"
        for i in range(control_width - 1)
    ]
    (output_root / "stimulus_feature_vocab.txt").write_text(
        "\n".join(vocab) + "\n")
    (output_root / "merge_schema.json").write_text(json.dumps({
        "sources": list(_REQUIRED_SOURCES),
        "source_roots": {
            name: _relative_reference(path, output_root)
            for name, path in source_paths.items()
        },
        "sample_id_prefixes": _SOURCE_PREFIX,
        "subject_policy": "source-qualified source-subject",
        "signal_modalities": ["calcium"],
        "control_modality": "stimulus",
        "control_width": control_width,
        "control_features": {
            "0": "source scalar drive; salt waveform or Randi opto waveform",
            "1..N": "Randi recording-local target gates; zero for salt/HF",
        },
        "missing_control_policy": (
            "HuggingFace rows omit stimulus; absence means no intervention"),
        "channel_identity_policy": (
            "source-local IDs; unified profile must not union-align"),
        "materialized": materialize,
        "path_policy": (
            "manifest relative references; materialize for a self-contained copy"
            if not materialize else "all arrays and companions copied into output"),
    }, indent=2, sort_keys=True) + "\n")
    print(
        f"[merge] {sum(counts.values())} samples -> {output_root} "
        f"(control_width={control_width}, materialized={materialize})")
    print(
        "[merge] sources: " + ", ".join(
            f"{name}={counts[name]} ({controls[name]} controls)"
            for name in _REQUIRED_SOURCES))
    return output_root / "manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toyoshima", required=True,
                        help="canonical Toyoshima/gKDR-GMM salt ladder")
    parser.add_argument("--randi", required=True,
                        help="canonical Randi PumpProbe ladder")
    parser.add_argument("--huggingface", required=True,
                        help="canonical HuggingFace activity ladder")
    parser.add_argument("--output", required=True,
                        help="unified manifest-federated ladder output root")
    parser.add_argument("--control-width", type=int, default=282)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--materialize", action="store_true",
        help="copy arrays into a self-contained output instead of references")
    args = parser.parse_args()
    merge_c_elegans_ladders(
        args.output,
        {
            "toyoshima": Path(args.toyoshima),
            "randi": Path(args.randi),
            "huggingface": Path(args.huggingface),
        },
        control_width=args.control_width,
        overwrite=args.overwrite,
        materialize=args.materialize,
    )


if __name__ == "__main__":
    main()
