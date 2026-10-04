#!/usr/bin/env python3
"""Generate recording-local TDE-RICA bases for the Randi PumpProbe corpus.

Randi channel IDs are recording-local, so this tool deliberately writes one
basis per recording instead of padding recordings into a false global channel
space. Invalid or constant channels are excluded from each fit and their
recording-local IDs are stored with the motif basis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tderica", required=True, type=Path)
    parser.add_argument("--dim-embed", type=int, default=12)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--fit-fraction", type=float, default=1.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-valid-channels", type=int, default=2)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_ids(path: Path, channels: int) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines()]
    if len(names) != channels or any(not name for name in names):
        raise ValueError(
            f"{path} has {len(names)} non-empty IDs; expected {channels}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate channel IDs")
    return names


def _fit_one(
    calcium_path: Path,
    mask_path: Path,
    ids_path: Path,
    output_path: Path,
    *,
    dim_embed: int,
    components: int,
    subsample: int,
    fit_fraction: float,
    lambda_: float,
    max_iter: int,
    min_valid_channels: int,
    decompose: Any,
) -> dict[str, Any]:
    calcium = np.asarray(np.load(calcium_path), dtype=np.float64)
    mask_present = mask_path.exists()
    mask = (
        np.asarray(np.load(mask_path), dtype=bool)
        if mask_present
        else np.ones_like(calcium, dtype=bool)
    )
    if calcium.ndim != 2 or mask.shape != calcium.shape:
        raise ValueError(
            f"expected calcium/mask (channels, frames), got "
            f"{calcium.shape} and {mask.shape}"
        )
    channels, frames = calcium.shape
    ids = _load_ids(ids_path, channels)
    fit_frames = min(frames, max(dim_embed + 3, int(round(frames * fit_fraction))))
    fit_frames = max(dim_embed + 3, fit_frames)
    fit_frames = min(frames, fit_frames)
    if fit_frames <= dim_embed + 2:
        raise ValueError(
            f"recording has only {frames} frames; fit requires more than "
            f"dim_embed+2={dim_embed + 2}"
        )

    finite = np.isfinite(calcium[:, :fit_frames])
    valid = mask[:, :fit_frames] & finite
    valid_channels = valid.all(axis=1)
    candidate = calcium[:, :fit_frames]
    variance = np.std(candidate, axis=1)
    nonconstant = variance > 1e-12
    keep = valid_channels & nonconstant
    kept_indices = np.flatnonzero(keep)
    if kept_indices.size < min_valid_channels:
        raise ValueError(
            f"only {kept_indices.size} fully-valid, nonconstant channels; "
            f"need at least {min_valid_channels}"
        )

    fit_data = candidate[kept_indices].T
    fit_components = min(components, fit_data.shape[0] - dim_embed + 1,
                         dim_embed * fit_data.shape[1])
    if fit_components < 1:
        raise ValueError("no feasible TDE-RICA component count")
    _, motifs, _ = decompose(
        fit_data[:, :, None],
        dim_embed=dim_embed,
        n_components=fit_components,
        subsample=subsample,
        whiten=False,
        lambda_=lambda_,
        max_iter=max_iter,
    )
    motifs = np.asarray(motifs, dtype=np.float64)
    if motifs.shape != (fit_components, dim_embed, kept_indices.size):
        raise ValueError(f"unexpected motif shape {motifs.shape}")
    if not np.isfinite(motifs).all():
        raise FloatingPointError("TDE-RICA produced non-finite motifs")

    kept_ids = [ids[int(index)] for index in kept_indices]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        coeffEmbed2=motifs,
        channel_ids=np.asarray(kept_ids),
        strNamesOrdered=np.asarray(kept_ids),
        channel_indices=kept_indices.astype(np.int64),
    )
    os.replace(temporary, output_path)
    return {
        "status": "ok",
        "recording_id": calcium_path.stem,
        "calcium_file": str(calcium_path),
        "mask_file": str(mask_path),
        "mask_present": bool(mask_present),
        "channel_ids_file": str(ids_path),
        "basis_file": str(output_path),
        "source_channels": int(channels),
        "source_frames": int(frames),
        "fit_frames": int(fit_frames),
        "fit_fraction_actual": float(fit_frames / frames),
        "valid_channels_before_constant_filter": int(valid_channels.sum()),
        "constant_channels_dropped": int((valid_channels & ~nonconstant).sum()),
        "basis_channels": int(kept_indices.size),
        "basis_components": int(fit_components),
        "embed_width": int(dim_embed),
        "subsample": int(subsample),
        "dt_s": 0.5,
        "motif_shape": [int(value) for value in motifs.shape],
        "finite": True,
    }


def main() -> int:
    args = _parse_args()
    if args.dim_embed < 2 or args.components < 1 or args.subsample < 1:
        raise ValueError("dim-embed, components, and subsample must be positive")
    if not 0 < args.fit_fraction <= 1:
        raise ValueError("fit-fraction must be in (0, 1]")
    if args.max_iter < 1 or args.min_valid_channels < 1:
        raise ValueError("max-iter and min-valid-channels must be positive")
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    tderica = args.tderica.resolve()
    calcium_dir = data_root / "calcium"
    mask_dir = data_root / "calcium_mask"
    ids_dir = data_root / "calcium_ids"
    calcium_files = sorted(
        calcium_dir.glob("*.npy"),
        key=lambda path: (
            (0, int(path.stem))
            if path.stem.isdigit()
            else (1, path.stem)
        ),
    )
    if not calcium_files:
        raise FileNotFoundError(f"no calcium/*.npy files under {data_root}")
    if args.max_recordings is not None:
        calcium_files = calcium_files[:args.max_recordings]

    if str(tderica) not in sys.path:
        sys.path.insert(0, str(tderica))
    from tderica import decompose

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        raw_records = previous.get("records", {})
        if isinstance(raw_records, dict):
            records = {
                str(record_id): record
                for record_id, record in raw_records.items()
                if isinstance(record, dict)
            }
        elif isinstance(raw_records, list):
            records = {
                str(record.get("recording_id")): record
                for record in raw_records
                if isinstance(record, dict) and record.get("recording_id") is not None
            }
        else:
            records = {}
    else:
        records = {}
    manifest = {
        "schema_version": 1,
        "status": "running",
        "source": "Randi et al. (2023) PumpProbe",
        "data_root": str(data_root),
        "basis_scope": "per_recording",
        "channel_identity": "recording_local",
        "channel_policy": "fully_valid_in_fit_scope_and_nonconstant",
        "fit_scope": "prefix_fraction" if args.fit_fraction < 1 else "full_recording",
        "fit_on_evaluation_window": bool(args.fit_fraction >= 1),
        "settings": {
            "dim_embed": int(args.dim_embed),
            "components_requested": int(args.components),
            "subsample": int(args.subsample),
            "fit_fraction": float(args.fit_fraction),
            "lambda": float(args.lambda_),
            "max_iter": int(args.max_iter),
            "min_valid_channels": int(args.min_valid_channels),
            "dt_s": 0.5,
            "input_orientation": "channels_by_frames",
            "input_scale": "ingest_randi_minmax_-3_3",
        },
        "records": records,
    }
    _atomic_json(manifest_path, manifest)

    failures = 0
    for index, calcium_path in enumerate(calcium_files, start=1):
        recording_id = calcium_path.stem
        output_path = output / f"randi_{recording_id}.npz"
        existing = records.get(recording_id)
        if args.resume and existing and existing.get("status") == "ok" and output_path.exists():
            print(f"[{index}/{len(calcium_files)}] skip {recording_id} (resume)", flush=True)
            continue
        print(f"[{index}/{len(calcium_files)}] fit {recording_id}", flush=True)
        try:
            record = _fit_one(
                calcium_path,
                mask_dir / calcium_path.name,
                ids_dir / calcium_path.name.replace(".npy", ".txt"),
                output_path,
                dim_embed=args.dim_embed,
                components=args.components,
                subsample=args.subsample,
                fit_fraction=args.fit_fraction,
                lambda_=args.lambda_,
                max_iter=args.max_iter,
                min_valid_channels=args.min_valid_channels,
                decompose=decompose,
            )
        except Exception as exc:  # keep independent recordings resumable
            failures += 1
            record = {
                "status": "failed",
                "recording_id": recording_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"[{index}/{len(calcium_files)}] FAILED {recording_id}: {exc}", flush=True)
        records[recording_id] = record
        manifest["records"] = records
        manifest["completed"] = int(sum(
            value.get("status") == "ok" for value in records.values()
        ))
        manifest["failed"] = int(sum(
            value.get("status") == "failed" for value in records.values()
        ))
        _atomic_json(manifest_path, manifest)

    manifest["status"] = "complete" if failures == 0 else "complete_with_failures"
    manifest["requested_recordings"] = int(len(calcium_files))
    manifest["completed"] = int(sum(
        value.get("status") == "ok" for value in records.values()
    ))
    manifest["failed"] = int(sum(
        value.get("status") == "failed" for value in records.values()
    ))
    _atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "requested_recordings": manifest["requested_recordings"],
        "completed": manifest["completed"],
        "failed": manifest["failed"],
        "output": str(output),
    }, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
