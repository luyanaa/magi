"""Ingest raw C. elegans calcium CSVs into the species ladder layout.

Source format (salt-stimulus recordings in ``cleandata_smoothened2``): per worm
``<id>_ratio.csv`` is a time-major matrix (T rows x N neurons, no header) of the
scaled YFP/CFP ratio of the YC2.60 cameleon, and ``<id>_uniqNames.csv`` lists
the N neuron identifiers (a mix of canonical names and recording-local numeric
tracker ids). Optional metadata makes the ladder scientifically usable:

* ``stimulation_timing.xlsx`` (gKDR-GMM / Toyoshima et al., PLOS Comput Biol
  2024) carries the **per-sample** frame rate (3.69-5.72 fps on the 24-worm
  pilot, i.e. never the 10 Hz some configs assume), the animal id, the
  anaesthesia flag and the salt-stimulus timing. Pass ``--timing-xlsx`` to use
  it; a single ``--rate`` is only a fallback and is never assumed silently.
* ``gKDR-GMM/metadata`` (``conneurons.csv`` / ``globalNames.csv``) names the
  canonical neurons when the identifiers are positional indices.

Channel selection follows the reference pipeline
(``gKDR-GMM/codes/make_common_data.m``): canonical-name membership
(``--filter-known``) plus an autocorrelation quality filter
(``--qc-autocorr``, default 0.3 at lag 20, which the reference uses to drop
noisy cells; on the pilot it removes ~46% of the named channels).

Output layout (consumed by ``SpeciesSignalDataset`` and
``train.py --data {"kind": "species", ...}``)::

    <output>/
        calcium/<id>.npy        # (N, T) float32 arrays
        calcium_ids/<id>.txt    # neuron identifiers (optional names)
        calcium_mask/<id>.npy   # present only when NaN cells exist
        stimulus/<id>.npy       # (1, T) salt drive (control role), when timed
        stimulus_trials/<id>.json  # [[t0, t1), ...] constant-concentration windows
        manifest.csv            # sample, subject(animal), session, origin,
                                # condition, rate_hz, dt_s, + provenance columns

Usage::

    python -m brain_moe_pinn.data.ingest_c_elegans \\
        --input <cleandata_smoothened2> --output <ladder_root> \\
        --timing-xlsx <cleandata_smoothened2/stimulation_timing.xlsx> \\
        --metadata <gKDR-GMM/metadata> --filter-known
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .readers import emit_sample
from .stimulus import SampleTiming, read_stimulation_timing, salt_waveform, trial_windows


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


def _lag_autocorrelation(matrix: np.ndarray, lag: int) -> np.ndarray:
    """Per-channel Pearson autocorrelation at ``lag`` frames (reference QC)."""
    if matrix.shape[1] <= lag + 1:
        return np.full(matrix.shape[0], np.nan)
    a, b = matrix[:, lag:], matrix[:, :-lag]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _select_channels(
    ids: Sequence[str],
    matrix: np.ndarray,
    mask: np.ndarray,
    *,
    pool: Optional[set],
    filter_known: bool,
    qc_threshold: Optional[float],
    qc_lag: int,
    sample_id: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, int]]:
    """Channel selection shared by the name filter and the reference QC.

    Selection is applied to the identifiers that will actually be written, so
    the name filter and the positional-name mapping cannot disagree (a mapped
    id is what ``--filter-known`` must test). An empty selection is an error,
    never a silently empty ``(0, T)`` sample.
    """
    keep = np.ones(len(ids), dtype=bool)
    dropped_unknown = 0
    if filter_known:
        if pool is None:
            raise ValueError(
                "--filter-known needs the canonical name pool; pass --metadata")
        known = np.array([token in pool for token in ids], dtype=bool)
        dropped_unknown = int((~known).sum())
        keep &= known
    dropped_noisy = 0
    if qc_threshold is not None:
        acf = _lag_autocorrelation(matrix, int(qc_lag))
        selected = keep & np.isfinite(acf) & (acf > float(qc_threshold))
        dropped_noisy = int((keep & ~selected).sum())
        keep = selected
    if not keep.any():
        raise ValueError(
            f"sample {sample_id!r}: channel selection kept 0 of {len(ids)} "
            f"channels (filter_known={filter_known}, "
            f"qc_autocorr={qc_threshold}); nothing to write")
    idx = np.flatnonzero(keep)
    return ([ids[i] for i in idx], matrix[idx], mask[idx],
            {"dropped_unknown": dropped_unknown, "dropped_noisy": dropped_noisy})


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    stimulus: str = "salt",
    limit: Optional[int] = None,
    metadata_dir: Optional[Path] = None,
    names_file: Optional[str] = "globalNames",
    apply_names: bool = False,
    filter_known: bool = False,
    rate_hz: Optional[float] = None,
    dt_s: Optional[float] = None,
    timing: Optional[Dict[int, SampleTiming]] = None,
    qc_autocorr: Optional[float] = 0.3,
    qc_lag: int = 20,
    stimulus_track: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert ``<id>_ratio.csv``/``<id>_uniqNames.csv`` pairs.

    Returns the manifest path. ``limit`` restricts the number of worms
    converted (smoke runs); the full directory is converted when None.
    Every sample needs a frame rate: from ``timing`` (per sample, keyed by the
    CSV index) when available, otherwise from the explicit ``rate_hz``.
    """
    if rate_hz is None and not timing:
        raise ValueError(
            "no frame rate available: pass --timing-xlsx (per-sample rates) or "
            "an explicit --rate. The salt recordings are NOT 10 Hz; a silent "
            "default would corrupt every time constant downstream.")
    input_dir = Path(input_dir)
    ratio_files = sorted(input_dir.glob("*_ratio.csv"),
                         key=lambda p: int(p.name.split("_")[0])
                         if p.name.split("_")[0].isdigit() else 0)
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

    totals = {"channels": 0, "kept": 0, "unknown": 0, "noisy": 0}
    for ratio_path in ratio_files:
        worm_id = ratio_path.name.replace("_ratio.csv", "")
        sample_index = int(worm_id) if worm_id.isdigit() else None
        names_path = ratio_path.with_name(f"{worm_id}_uniqNames.csv")
        raw_ids = _read_names(names_path)
        data = _read_ratio_csv(ratio_path)
        # CSVs are time-major (T, N); the ladder expects (C, T).
        matrix = np.ascontiguousarray(data.T, dtype=np.float32)
        mask = np.isfinite(matrix)
        clean = np.nan_to_num(matrix, nan=0.0)

        sample_timing = timing.get(sample_index) if (
            timing and sample_index is not None) else None
        if sample_timing is not None:
            sample_rate = float(sample_timing.fps)
            sample_dt = sample_timing.dt_s
        else:
            sample_rate = float(rate_hz)
            sample_dt = float(dt_s) if dt_s is not None else 1.0 / sample_rate

        ids: List[str] = list(raw_ids) if raw_ids else []
        extra: Dict[str, str] = {}
        if sample_timing is not None:
            extra = {
                "animal": str(sample_timing.animal),
                "sample_name": sample_timing.name,
                "anesthesia": str(sample_timing.anesthesia),
                "fps": f"{sample_timing.fps:.6f}",
                "stimulus_first_frame": f"{sample_timing.first_stimulus_frame:.3f}",
                "stimulus_first_s": f"{sample_timing.first_stimulus_s:.3f}",
                "stimulus_period_frames": f"{sample_timing.switch_period_frames:.3f}",
                "stimulus_period_s": f"{sample_timing.switch_period_s:.3f}",
                "stimulus_cycles": f"{sample_timing.cycles:.3f}",
                "duration_s": f"{sample_timing.total_duration_s:.3f}",
                "frames": str(int(matrix.shape[1])),
            }
        if raw_ids and apply_names:
            mapped = _map_positional_names(raw_ids, name_pool)
            if mapped is None:
                print(f"[ingest] worm {worm_id}: ids not positional "
                      f"({raw_ids[:3]}...); keeping raw identifiers")
            else:
                ids = mapped

        dropped = {"dropped_unknown": 0, "dropped_noisy": 0}
        if ids:
            if len(ids) != matrix.shape[0]:
                raise ValueError(
                    f"worm {worm_id}: {matrix.shape[0]} channels but "
                    f"{len(ids)} identifiers")
            ids, clean, mask, dropped = _select_channels(
                ids, clean, mask, pool=known_names, filter_known=filter_known,
                qc_threshold=qc_autocorr, qc_lag=qc_lag, sample_id=worm_id)
            if dropped["dropped_unknown"] or dropped["dropped_noisy"]:
                print(f"[ingest] worm {worm_id}: kept {len(ids)} channels "
                      f"(-{dropped['dropped_unknown']} non-canonical, "
                      f"-{dropped['dropped_noisy']} autocorr<={qc_autocorr})")
        totals["channels"] += (len(ids) + dropped["dropped_unknown"]
                               + dropped["dropped_noisy"])
        totals["kept"] += len(ids)
        totals["unknown"] += dropped["dropped_unknown"]
        totals["noisy"] += dropped["dropped_noisy"]

        signals: Dict[str, np.ndarray] = {"calcium": clean}
        masks: Dict[str, np.ndarray] = {} if mask.all() else {"calcium": mask}
        trials: Dict[str, Sequence[Tuple[int, int]]] = {}
        if stimulus_track and sample_timing is not None:
            n_frames = int(matrix.shape[1])
            signals["stimulus"] = salt_waveform(
                n_frames, sample_timing.first_stimulus_frame,
                sample_timing.switch_period_frames).reshape(1, -1)
            windows = trial_windows(
                n_frames, sample_timing.first_stimulus_frame,
                sample_timing.switch_period_frames)
            if windows:
                trials["stimulus"] = windows

        emit_sample(
            Path(output_dir), worm_id,
            signals,
            subject=(str(sample_timing.animal) if sample_timing is not None
                     else worm_id),
            session="1",
            origin=input_dir.name,
            condition=stimulus,
            rate_hz=sample_rate,
            dt_s=sample_dt,
            ids={"calcium": ids} if ids else None,
            masks=masks or None,
            trials=trials or None,
            overwrite=overwrite,
            extra=extra or None,
        )

    manifest_path = Path(output_dir) / "manifest.csv"
    print(f"[ingest] {len(ratio_files)} worms -> {calcium_dir} "
          f"(manifest: {manifest_path})")
    print(f"[ingest] channels: {totals['kept']} kept of {totals['channels']} "
          f"(dropped {totals['unknown']} non-canonical, "
          f"{totals['noisy']} below autocorr QC)")
    if totals["unknown"] == totals["channels"]:
        print("[ingest] WARNING: every channel was dropped; check --filter-known "
              "against the identifier semantics of this corpus")
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
    parser.add_argument("--timing-xlsx", default=None,
                        help="stimulation_timing.xlsx: per-sample frames/sec, "
                             "animal, anaesthesia and salt-stimulus timing")
    parser.add_argument("--rate", type=float, default=None,
                        help="single fallback rate in Hz for every sample; "
                             "required when --timing-xlsx is not given")
    parser.add_argument("--qc-autocorr", type=float, default=0.3,
                        help="reference channel QC: drop channels whose "
                             "autocorrelation at --qc-lag is not above this "
                             "(gKDR-GMM make_common_data.m); 0 disables")
    parser.add_argument("--qc-lag", type=int, default=20,
                        help="lag in frames for the autocorrelation QC")
    parser.add_argument("--no-stimulus-track", action="store_true",
                        help="do not write the salt drive / trial windows")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing sample arrays in the output")
    args = parser.parse_args()

    timing = (read_stimulation_timing(args.timing_xlsx)
              if args.timing_xlsx else None)
    if timing:
        rates = sorted({round(t.fps, 4) for t in timing.values()})
        print(f"[ingest] timing table: {len(timing)} samples, "
              f"frames/sec in [{rates[0]}, {rates[-1]}]")
    convert_directory(
        Path(args.input), Path(args.output),
        stimulus=args.stimulus, limit=args.limit,
        metadata_dir=Path(args.metadata) if args.metadata else None,
        names_file=args.names_file, apply_names=args.apply_names,
        filter_known=args.filter_known, rate_hz=args.rate, timing=timing,
        qc_autocorr=(None if args.qc_autocorr <= 0 else args.qc_autocorr),
        qc_lag=args.qc_lag,
        stimulus_track=not args.no_stimulus_track,
        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
