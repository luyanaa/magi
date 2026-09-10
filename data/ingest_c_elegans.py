"""Ingest raw C. elegans calcium CSVs into the species ladder layout.

Source format (salt-stimulus recordings in ``cleandata_smoothened2``): per
worm ``<id>_ratio.csv`` is a time-major matrix (T rows x N neurons, no
header) of normalized fluorescence ratios, and ``<id>_uniqNames.csv`` lists
the N neuron identifiers. Optional gKDR-GMM metadata
(``gKDR-GMM/metadata``: ``conneurons.csv`` / ``globalNames.csv``) can be
used to name neurons when the identifiers are positional indices.

Output layout (consumed by ``SpeciesSignalDataset`` and
``train.py --data {"kind": "species", ...}``)::

    <output>/
        calcium/<id>.npy        # (N, T) float32 arrays
        calcium_ids/<id>.txt    # neuron identifiers (optional names)
        calcium_mask/<id>.npy   # present only when NaN cells exist
        manifest.csv            # sample, subject, origin, condition, rate

Usage::

    python -m brain_moe_pinn.data.ingest_c_elegans \\
        --input <cleandata_smoothened2> --output <ladder_root> \\
        [--stimulus salt] [--limit N] [--metadata <gKDR-GMM/metadata>]
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .readers import emit_sample


def _read_ratio_csv(path: Path) -> np.ndarray:
    """Load a (T, N) time-major ratio matrix; NaN cells are preserved."""
    data = np.genfromtxt(path, delimiter=",", dtype=np.float64, missing_values="")
    if data.ndim != 2:
        raise ValueError(f"{path} is not a 2-D (T, N) matrix")
    return data


def _read_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _metadata_names(metadata_dir: Optional[Path], choose: str = "globalNames"):
    if metadata_dir is None:
        return None
    candidates = ("globalNames.csv", "conneurons.csv")
    if choose not in ("globalNames", "conneurons"):
        raise ValueError("choose must be globalNames or conneurons")
    fpath = Path(metadata_dir) / f"{choose}.csv"
    return _read_names(fpath)


def _known_name_pool(metadata_dir: Optional[Path]) -> Optional[set]:
    """Union of canonical neuron names from the gKDR-GMM metadata files."""
    if metadata_dir is None:
        return None
    pool: set = set()
    found = False
    for name in ("globalNames.csv", "conneurons.csv"):
        fpath = Path(metadata_dir) / name
        if fpath.exists():
            found = True
            pool.update(_read_names(fpath) or [])
    return pool if found else None


def _filter_known(ids: List[str], matrix: np.ndarray, mask: np.ndarray,
                  pool: set):
    """Keep only channels whose id is a canonical name in ``pool``.

    The salt corpus mixes canonical neuron names with recording-local
    numeric ids; alignment across worms requires the canonical subset.
    Returns (ids, matrix, mask, dropped_count).
    """
    keep = [i for i, token in enumerate(ids) if token in pool]
    if len(keep) == len(ids):
        return ids, matrix, mask, 0
    idx = np.asarray(keep, dtype=int)
    return ([ids[i] for i in keep], matrix[idx], mask[idx],
            len(ids) - len(keep))


def _map_positional_names(ids: List[str],
                          names: Optional[List[str]]) -> Optional[List[str]]:
    """Map numeric positional identifiers to canonical names (1-based).

    Applies only when every identifier parses as an int within the name
    list; otherwise returns None (caller keeps raw identifiers).
    """
    if names is None:
        return None
    mapped = []
    for token in ids:
        try:
            idx = int(token)
        except ValueError:
            return None
        if not (1 <= idx <= len(names)):
            return None
        mapped.append(names[idx - 1])
    return mapped


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    stimulus: str = "salt",
    limit: Optional[int] = None,
    metadata_dir: Optional[Path] = None,
    names_file: Optional[str] = "globalNames",
    apply_names: bool = False,
    filter_known: bool = False,
    rate_hz: float = 10.0,
    dt_s: Optional[float] = None,
) -> Path:
    """Convert ``<id>_ratio.csv``/``<id>_uniqNames.csv`` pairs.

    Returns the manifest path. ``limit`` restricts the number of worms
    converted (smoke runs); the full directory is converted when None.
    """
    input_dir = Path(input_dir)
    ratio_files = sorted(input_dir.glob("*_ratio.csv"))
    if not ratio_files:
        raise FileNotFoundError(f"no *_ratio.csv files under {input_dir}")
    if limit is not None:
        ratio_files = ratio_files[:max(1, int(limit))]

    calcium_dir = Path(output_dir) / "calcium"
    calcium_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_dir) if metadata_dir else None
    name_pool = _metadata_names(metadata_path, names_file)
    known_names = _known_name_pool(metadata_path)
    if name_pool and apply_names:
        print(f"[ingest] mapping positional ids via {names_file} "
              f"({len(name_pool)} names)")
    elif known_names is not None:
        print(f"[ingest] metadata dir given; keeping raw identifiers "
              f"({len(known_names)} canonical names available for "
              f"--filter-known)")

    for ratio_path in ratio_files:
        worm_id = ratio_path.name.replace("_ratio.csv", "")
        names_path = ratio_path.with_name(f"{worm_id}_uniqNames.csv")
        raw_ids = _read_names(names_path)
        data = _read_ratio_csv(ratio_path)
        # CSVs are time-major (T, N); the ladder expects (C, T).
        matrix = np.ascontiguousarray(data.T, dtype=np.float32)
        mask = np.isfinite(matrix)
        clean = np.nan_to_num(matrix, nan=0.0)
        ids: Optional[List[str]] = raw_ids
        if raw_ids and apply_names:
            mapped = _map_positional_names(raw_ids, name_pool)
            if mapped is None:
                print(f"[ingest] worm {worm_id}: ids not positional "
                      f"({raw_ids[:3]}...); keeping raw identifiers")
            else:
                ids = mapped
        if raw_ids and filter_known and known_names is not None:
            ids, clean, mask, dropped = _filter_known(
                raw_ids, clean, mask, known_names)
            if dropped:
                print(f"[ingest] worm {worm_id}: dropped {dropped} "
                      f"non-canonical channels")
        emit_sample(
            Path(output_dir), worm_id,
            {"calcium": clean},
            subject=worm_id,
            session="1",
            origin=input_dir.name,
            condition=stimulus,
            rate_hz=rate_hz,
            dt_s=dt_s,
            ids={"calcium": ids} if ids else None,
            masks={"calcium": mask} if not mask.all() else None,
            overwrite=False,
        )

    manifest_path = Path(output_dir) / "manifest.csv"
    print(f"[ingest] {len(ratio_files)} worms -> {calcium_dir} "
          f"(manifest: {manifest_path})")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="directory of <id>_ratio.csv files")
    parser.add_argument("--output", required=True,
                        help="species ladder root (calcium/ + manifest.csv)")
    parser.add_argument("--stimulus", default="salt")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert at most N worms (smoke runs)")
    parser.add_argument("--metadata", default=None,
                        help="gKDR-GMM/metadata dir (conneurons/globalNames)")
    parser.add_argument("--names-file", choices=("globalNames", "conneurons"),
                        default="globalNames")
    parser.add_argument("--apply-names", action="store_true",
                        help="map positional ids to canonical names")
    parser.add_argument("--filter-known", action="store_true",
                        help="keep only channels whose id is in the "
                             "gKDR-GMM metadata name pool (enables "
                             "cross-worm name alignment)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="recording rate in Hz (default 10)")
    args = parser.parse_args()
    convert_directory(
        Path(args.input), Path(args.output),
        stimulus=args.stimulus, limit=args.limit,
        metadata_dir=Path(args.metadata) if args.metadata else None,
        names_file=args.names_file, apply_names=args.apply_names,
        filter_known=args.filter_known, rate_hz=args.rate)


if __name__ == "__main__":
    main()
