"""Species ladder datasets: manifest-driven, subject-grouped, channel-aware.

Contract consumed by the generic species training path
(``BrainMoEPINN.forward_modalities`` + ``TotalLoss.recon_extra``): every
batch is a dict of modality -> stacked tensors plus optional masks and
metadata keys. Design follows the corpus survey in ``docs/data_corpus_plan.md``
§6 (P0/P1/P2): subject/session hierarchy, per-sample channel identity and
union alignment, per-channel masks, seconds-based windows, condition/region/
origin metadata, trial windows, static graphs, and subject-grouped splits.

Canonical layout (all optional per feature)::

    root/
        manifest.csv                    # sample_id,subject,session,origin,
                                        # condition,rate_hz,dt_s[,<mod>_file...]
        <modality>/<sample>.npy         # (C, T) float signal
        <modality>_ids/<sample>.txt     # one channel id per line
        <modality>_mask/<sample>.npy    # bool (C, T) valid-channel/frame mask
        <modality>_trials/<sample>.json # [[t0, t1], ...] trial windows
        graphs/<sample>.npy             # static (C, C) adjacency (structure)
        regions.csv                     # channel_id,region  (load-time pooling)

Without a manifest, an index is inferred from the modality directories
(sample = file stem); federation (per-subject modality absence) is
supported by indexing the union of stems.
"""

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_FORMATS = (".npy", ".npz", ".pth")
_STR_KEYS = ("sample_id", "subject", "session", "origin", "condition")
_META_KEYS = _STR_KEYS + ("dt",)
# Companion directories of the canonical layout. Modality-scoped companions
# are prefixed; the static graph is a single shared directory (``graphs/``),
# not ``graph_graphs/``.
_COMPANION_DIRS = {
    "ids": "{modality}_ids",
    "mask": "{modality}_mask",
    "trials": "{modality}_trials",
    "graphs": "graphs",
}


def _load_array(fpath: Path) -> torch.Tensor:
    if fpath.suffix == ".npy":
        data = np.load(fpath)
    elif fpath.suffix == ".npz":
        data = np.load(fpath)["data"]
    elif fpath.suffix == ".pth":
        data = torch.load(fpath, weights_only=True)
        if hasattr(data, "numpy"):
            data = data.numpy()
        return torch.as_tensor(data).float()
    else:
        raise ValueError(f"Unsupported format: {fpath}")
    return torch.from_numpy(data).float()


def _read_ids(fpath: Optional[Path]) -> Optional[List[str]]:
    if fpath is None or not fpath.exists():
        return None
    return [line.strip() for line in fpath.read_text().splitlines() if line.strip()]


def _read_trials(fpath: Optional[Path]) -> Optional[List[Tuple[int, int]]]:
    if fpath is None or not fpath.exists():
        return None
    raw = json.loads(fpath.read_text())
    return [(int(a), int(b)) for a, b in raw]


def _region_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None or not path.exists():
        return None
    mapping = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            channel = row.get("channel_id")
            region = row.get("region")
            if channel is not None and region is not None:
                mapping[channel.strip()] = region.strip()
    return mapping


def _aggregate_regions(
    signal: torch.Tensor,
    mask: torch.Tensor,
    ids: List[str],
    regions: List[str],
    region_of: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean signal per region over its channels; mask = any channel valid."""
    import collections
    groups = collections.OrderedDict((r, []) for r in regions)
    for i, cid in enumerate(ids):
        r = region_of.get(cid)
        if r is not None and r in groups:
            groups[r].append(i)
    out_rows, out_mask = [], []
    for r in regions:
        rows = groups[r]
        if not rows:
            out_rows.append(torch.zeros(signal.shape[-1], dtype=signal.dtype,
                                        device=signal.device))
            out_mask.append(torch.zeros(signal.shape[-1], dtype=torch.bool,
                                        device=signal.device))
            continue
        idx = torch.tensor(rows, dtype=torch.long)
        sub = signal[idx]
        sub_mask = mask[idx]
        denom = sub_mask.float().sum(dim=0).clamp_min(1.0)
        mean = (sub * sub_mask.float()).sum(dim=0) / denom
        out_rows.append(mean)
        out_mask.append(sub_mask.any(dim=0))
    return (torch.stack(out_rows), torch.stack(out_mask))


def read_manifest(path: Optional[Path]) -> List[Dict[str, str]]:
    """Read a canonical manifest CSV; returns one dict per row."""
    if path is None or not Path(path).exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


class SpeciesSignalDataset(Dataset):
    """Manifest-driven multi-modality signal dataset.

    Rows are sample units (subject x session). Each row may carry any subset
    of the requested modalities (federation); windowing is per-sample;
    masks, channel ids, trials, and static graphs are optional features.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        modalities: Sequence[str],
        seq_len: Optional[int] = None,
        seq_seconds: Optional[float] = None,
        manifest: Union[str, Path, None] = "manifest.csv",
        rows: Optional[Iterable[Dict[str, str]]] = None,
        load_ids: bool = False,
        align_channels: bool = False,
        max_union_channels: int = 4096,
        region_map: Union[str, Path, None] = None,
        load_graphs: bool = False,
        return_masks: bool = True,
        use_trials: bool = True,
        rate_default: Optional[float] = None,
        random_windows: bool = True,
        return_next_step_targets: bool = False,
        future_steps: int = 1,
        roles: Optional[Dict[str, str]] = None,
        species: Optional[str] = None,
        rate_tolerance: float = 1e-3,
        strict_rate: bool = True,
        normalization: Optional[str] = None,
        normalization_stats: Optional[
            Mapping[str, Mapping[str, float]]] = None,
        rng_seed: int = 0,
        expected_rate_hz: Optional[float] = None,
    ):
        if not modalities:
            raise ValueError("at least one modality is required")
        if seq_len is None and seq_seconds is None:
            seq_len = 256
        if (seq_len is None) == (seq_seconds is None):
            raise ValueError("exactly one of seq_len (frames) or seq_seconds must be given")
        if seq_len is not None and seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_seconds is not None and seq_seconds <= 0:
            raise ValueError("seq_seconds must be positive")
        self.root = Path(data_dir)
        self.modalities = tuple(modalities)
        self.seq_len = seq_len
        self.seq_seconds = seq_seconds
        self.load_ids = bool(load_ids)
        self.align_channels = bool(align_channels)
        self.max_union_channels = int(max_union_channels)
        self.load_graphs = bool(load_graphs)
        self.return_masks = bool(return_masks)
        self.use_trials = bool(use_trials)
        self.rate_default = rate_default
        self.random_windows = bool(random_windows)
        self.return_next_step_targets = bool(return_next_step_targets)
        self.rng = torch.Generator().manual_seed(int(rng_seed))
        self.future_steps = int(future_steps)
        if self.future_steps < 1:
            raise ValueError("future_steps must be positive")
        roles = dict(roles or {})
        invalid_roles = set(roles.values()) - {"signal", "control", "aux", "graph"}
        if invalid_roles:
            raise ValueError(f"unknown modality roles: {sorted(invalid_roles)}")
        unknown_role_modalities = set(roles) - set(self.modalities)
        if unknown_role_modalities:
            raise ValueError(
                f"roles reference modalities not requested: "
                f"{sorted(unknown_role_modalities)}")
        self.roles = roles
        self.species = species
        self.expected_rate_hz = (
            None if expected_rate_hz is None else float(expected_rate_hz))
        if self.expected_rate_hz is not None and self.expected_rate_hz <= 0:
            raise ValueError("expected_rate_hz must be positive")
        self.rate_tolerance = float(rate_tolerance)
        self.strict_rate = bool(strict_rate)
        self.normalization = str(normalization or "none").lower()
        if self.normalization not in {"none", "global_zscore"}:
            raise ValueError(
                "normalization must be 'none' or 'global_zscore'")
        self.normalization_stats = {
            str(modality): {
                "mean": float(values["mean"]),
                "std": max(float(values["std"]), 1e-6),
            }
            for modality, values in (normalization_stats or {}).items()
        }
        self.regions = _region_map(Path(region_map) if region_map else None)
        self._mod_files: Dict[str, Path] = {}

        for modality in self.modalities:
            d = self.root / modality
            if d.is_dir():
                self._mod_files[modality] = d

        self.manifest_path = None
        if manifest is not None:
            mp = Path(manifest)
            if not mp.is_absolute():
                mp = self.root / mp
            if mp.exists():
                self.manifest_path = mp

        manifest_rows = read_manifest(self.manifest_path)
        if rows is None:
            self.rows = manifest_rows
            if not self.rows:
                self.rows = self._infer_rows()
        else:
            # An explicit empty split is a valid loader (e.g. val_frac=0).
            # Do not silently repopulate it from every modality directory.
            self.rows = list(rows)

        # Channel identity / union index / region space, computed once.
        self._ids: Dict[str, Optional[List[str]]] = {}
        self._global_ids: Optional[List[str]] = None
        self._regions: Optional[List[str]] = None
        if self.load_ids or self.align_channels or self.regions is not None:
            self._index_channels()

    # ------------------------------------------------------------------ rows
    def _infer_rows(self) -> List[Dict[str, str]]:
        present = [d for d in self._mod_files.values() if d.is_dir()]
        if not present:
            raise FileNotFoundError(
                f"no modality directories under {self.root}; "
                f"wanted {self.modalities}")
        stems = []
        for d in present:
            stems.extend(
                p.stem for p in d.glob("*")
                if p.is_file() and p.suffix in _FORMATS)
        return [{"sample_id": s, "subject": s, "session": "1"}
                for s in sorted(set(stems))]

    def _row_file(self, row: Dict[str, str], modality: str,
                  suffix_dir: Optional[str] = None,
                  ext: str = ".npy") -> Optional[Path]:
        col = f"{modality}_file" if suffix_dir is None else None
        if suffix_dir is not None:
            if suffix_dir not in _COMPANION_DIRS:
                raise ValueError(
                    f"unknown companion directory {suffix_dir!r}; expected one "
                    f"of {sorted(_COMPANION_DIRS)}")
            companion_keys = [f"{modality}_{suffix_dir}_file"]
            if suffix_dir == "graphs":
                companion_keys.append("graphs_file")
            for companion_key in companion_keys:
                override = row.get(companion_key)
                if override:
                    p = Path(override)
                    return p if p.is_absolute() else (self.root / p)
            base = self.root / _COMPANION_DIRS[suffix_dir].format(
                modality=modality)
            candidate = base / f"{row.get('sample_id', '')}{ext}"
        else:
            override = row.get(col)
            if override:
                p = Path(override)
                return p if p.is_absolute() else (self.root / p)
            candidate = self._mod_files.get(modality) and (
                self._mod_files[modality] / f"{row.get('sample_id', '')}{ext}")
        return candidate if candidate and candidate.is_file() else None

    def _sample_ids(self, row: Dict[str, str],
                    modality: str) -> Optional[List[str]]:
        fpath = self._row_file(row, modality, suffix_dir="ids", ext=".txt")
        return _read_ids(fpath)

    def _channel_index(self) -> None:
        # union of per-sample ids over all rows (only when ids requested)
        if self._global_ids is not None:
            return
        union: List[str] = []
        seen = set()
        for row in self.rows:
            for modality in self.modalities:
                ids = self._sample_ids(row, modality)
                if not ids:
                    continue
                for cid in ids:
                    if cid not in seen:
                        seen.add(cid)
                        union.append(cid)
        if len(union) > self.max_union_channels:
            raise ValueError(
                f"channel union has {len(union)} channels > max_union_channels "
                f"{self.max_union_channels}; use region_map or raw per-sample "
                f"channels with batch-1")
        self._global_ids = union

    def _index_channels(self) -> None:
        if self.regions is not None:
            regions = sorted(set(self.regions.values()))
            self._regions = regions
        if self.align_channels:
            self._channel_index()

    def _rate_hz(self, row: Dict[str, str], modality: str) -> float:
        """Return the modality-specific sampling rate for one manifest row.

        When ``expected_rate_hz`` is set (the species profile declares a
        single-rate contract), the manifest value must agree: a mismatch means
        every time constant and physical window derived downstream would be
        computed with the wrong clock, so it is an error rather than a silent
        preference for one of the two.
        """
        value = row.get(f"{modality}_rate_hz") or row.get(
            f"{modality}_hz")
        explicit = bool(value)
        dt = row.get(f"{modality}_dt_s")
        if value:
            rate = float(value)
        elif dt:
            rate = 1.0 / float(dt)
        else:
            base = row.get("rate_hz") or self.rate_default
            explicit = bool(base)
            rate = float(base) if base else 1.0
        if rate <= 0:
            raise ValueError(
                f"sampling rate for modality {modality!r} must be positive")
        if (self.expected_rate_hz is not None and self.strict_rate
                and explicit):
            rel = abs(rate - self.expected_rate_hz) / self.expected_rate_hz
            if rel > self.rate_tolerance:
                raise ValueError(
                    f"sampling-rate mismatch for sample "
                    f"{row.get('sample_id', '?')!r} modality {modality!r}: "
                    f"manifest says {rate:.6g} Hz but the species profile "
                    f"declares {self.expected_rate_hz:.6g} Hz. Either point the "
                    f"profile at the corpus rate, set sample_rate_hz_source="
                    f"\"manifest\" for per-sample rates, or set "
                    f"strict_rate=False to accept the manifest clock.")
        return rate

    def _frame_count(self, row: Dict[str, str], modality: str) -> int:
        if self.seq_len is not None:
            return int(self.seq_len)
        rate = self._rate_hz(row, modality)
        return max(1, int(round(rate * float(self.seq_seconds))))

    def _allowed_starts(self, length: int, trials: Optional[List[Tuple[int, int]]],
                        win: int) -> List[int]:
        if win <= 0 or length < win:
            return []
        if not trials or not self.use_trials:
            return list(range(0, length - win + 1))
        starts = set()
        for lo, hi in trials:
            a = max(lo, 0)
            b = min(hi, length)
            if b - a >= win:
                starts.update(range(a, b - win + 1))
        return sorted(starts)

    def _pick_start_time(self, start_times: List[float]) -> float:
        if not start_times:
            return 0.0
        if not self.random_windows:
            return float(start_times[0])
        idx = int(torch.randint(
            len(start_times), (1,), generator=self.rng).item())
        return float(start_times[idx])

    @staticmethod
    def _slice_window(
        data: torch.Tensor, start: int, win: int

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice and zero-pad a window, returning its validity mask."""
        channels = data.shape[0]
        window = torch.zeros(
            channels, win, dtype=data.dtype, device=data.device)
        valid = torch.zeros(
            channels, win, dtype=torch.bool, device=data.device)
        source_start = max(0, int(start))
        if source_start >= data.shape[1]:
            return window, valid
        copied = min(win, data.shape[1] - source_start)
        window[:, :copied] = data[:, source_start:source_start + copied]
        valid[:, :copied] = True
        return window, valid

    def _normalize_window(
        self,
        modality: str,
        window: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply training-fitted global statistics without touching padding."""
        if self.normalization != "global_zscore":
            return window
        if self.roles.get(modality, "signal") != "signal":
            return window
        stats = self.normalization_stats.get(modality)
        if stats is None:
            return window
        valid = mask & torch.isfinite(window)
        mean = window.new_tensor(stats["mean"])
        std = window.new_tensor(stats["std"])
        normalized = (window - mean) / std
        return torch.where(valid, normalized, torch.zeros_like(window))

    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        sample_id = row.get("sample_id", f"row{idx}")
        out: Dict[str, Any] = {}
        modality_info = []
        for modality in self.modalities:
            fpath = self._row_file(row, modality)
            if fpath is None:
                continue
            data = _load_array(fpath)
            if data.dim() != 2:
                raise ValueError(f"{fpath} is not a (C, T) array")
            rate = self._rate_hz(row, modality)
            win = self._frame_count(row, modality)
            trials = _read_trials(self._row_file(
                row, modality, suffix_dir="trials", ext=".json"))
            if self.roles.get(modality) == "control":
                # Control tracks are cut at signal-window starts; their trial
                # metadata describes stimulus segments, not valid starts.
                trials = None
            required = (
                win * (1 + self.future_steps)
                if self.return_next_step_targets else win)
            starts = self._allowed_starts(data.shape[1], trials, required)
            if not starts:
                if self.return_next_step_targets:
                    raise ValueError(
                        f"sample {sample_id!r}, modality {modality!r} has "
                        f"no context+{self.future_steps}-horizon window of "
                        f"{win} frames")
                starts = [0]
            modality_info.append({
                "modality": modality,
                "fpath": fpath,
                "data": data,
                "rate": rate,
                "win": win,
                "starts": starts,
                "start_set": set(starts),
            })

        if not modality_info:
            for key in _STR_KEYS:
                if row.get(key):
                    out[key] = row[key]
            return out

        if self.seq_len is not None and len({
            info["rate"] for info in modality_info
        }) > 1:
            raise ValueError(
                "seq_len is frame-based and cannot align modalities with "
                "different sampling rates; use seq_seconds")

        # Candidate starts are represented in seconds so modalities with
        # different clocks still select physically aligned windows.
        reference = modality_info[0]
        candidate_times = [
            start / reference["rate"] for start in reference["starts"]]
        valid_times = []
        starts_by_time = []
        for time_s in candidate_times:
            starts_for_time = {}
            valid = True
            for info in modality_info:
                start = int(round(time_s * info["rate"]))
                if start not in info["start_set"]:
                    valid = False
                    break
                starts_for_time[info["modality"]] = start
            if valid:
                valid_times.append(time_s)
                starts_by_time.append(starts_for_time)
        if not valid_times:
            raise ValueError(
                f"modalities in sample {sample_id!r} have no synchronized "
                "physical-time windows")
        chosen = self._pick_start_time(valid_times)
        chosen_idx = min(
            range(len(valid_times)),
            key=lambda i: abs(valid_times[i] - chosen))
        starts_for_time = starts_by_time[chosen_idx]

        for info in modality_info:
            modality = info["modality"]
            data = info["data"]
            start = starts_for_time[modality]
            win = info["win"]
            window, valid = self._slice_window(data, start, win)
            mask_path = self._row_file(row, modality, suffix_dir="mask")
            mask_data = None
            if mask_path is not None:
                mask_data = _load_array(mask_path)
                if mask_data.shape == data.shape:
                    mask_window, mask_valid = self._slice_window(
                        mask_data, start, win)
                    valid = valid & mask_valid & mask_window.bool()
                else:
                    warnings.warn(
                        f"sample {sample_id!r} modality {modality!r}: mask "
                        f"{tuple(mask_data.shape)} does not match signal "
                        f"{tuple(data.shape)}; validity information is being "
                        "ignored for this sample", RuntimeWarning)
                    mask_data = None
            mask = valid

            future_windows = []
            if self.return_next_step_targets:
                for horizon in range(1, self.future_steps + 1):
                    future_window, future_valid = self._slice_window(
                        data, start + horizon * win, win)
                    if mask_data is not None:
                        next_mask_window, next_mask_valid = (
                            self._slice_window(
                                mask_data, start + horizon * win, win))
                        future_valid = (
                            future_valid & next_mask_valid
                            & next_mask_window.bool())
                    future_windows.append((future_window, future_valid))

            window = self._normalize_window(modality, window, mask)
            future_windows = [
                (self._normalize_window(modality, future, future_mask),
                 future_mask)
                for future, future_mask in future_windows
            ]

            ids = self._sample_ids(row, modality) if (
                self.load_ids or self.align_channels or
                self.regions is not None) else None
            if ids is not None and len(ids) != window.shape[0]:
                warnings.warn(
                    f"sample {sample_id!r} modality {modality!r}: "
                    f"{len(ids)} channel ids for {window.shape[0]} channels; "
                    "channel identity is unavailable for this sample "
                    "(union alignment will skip it)", RuntimeWarning)
                ids = None
            if self.regions is not None and ids is not None:
                window, mask = _aggregate_regions(
                    window, mask, ids, self._regions, self.regions)
                future_windows = [
                    _aggregate_regions(
                        future, future_mask, ids, self._regions, self.regions)
                    for future, future_mask in future_windows
                ]
                ids = self._regions
            if self.align_channels and ids is not None:
                source_ids = list(ids)
                window, mask, ids = self._align_to_union(
                    window, mask, source_ids, sample_id, modality)
                aligned_future_windows = []
                for future, future_mask in future_windows:
                    aligned_future_windows.append(self._align_to_union(
                        future, future_mask, source_ids, sample_id, modality)[:2])
                future_windows = aligned_future_windows

            out[modality] = window
            if self.return_masks:
                out[f"{modality}_mask"] = mask
            if self.load_ids:
                out[f"{modality}_ids"] = ids or []
            if future_windows:
                if self.future_steps == 1:
                    out[f"{modality}_next"] = future_windows[0][0]
                    if self.return_masks:
                        out[f"{modality}_next_mask"] = future_windows[0][1]
                else:
                    out[f"{modality}_future"] = torch.stack(
                        [future for future, _ in future_windows], dim=0)
                    if self.return_masks:
                        out[f"{modality}_future_mask"] = torch.stack(
                            [future_mask
                             for _, future_mask in future_windows], dim=0)
                    for horizon, (future, future_mask) in enumerate(
                            future_windows, start=1):
                        out[f"{modality}_next_{horizon}"] = future
                        if self.return_masks:
                            out[f"{modality}_next_{horizon}_mask"] = future_mask

        if self.load_graphs:
            graph_path = self._row_file(
                row, "", suffix_dir="graphs", ext=".npy")
            if graph_path is not None:
                graph = _load_array(graph_path)
                if graph.dim() != 2 or graph.shape[0] != graph.shape[1]:
                    raise ValueError(
                        f"{graph_path} must be a square graph adjacency matrix")
                out["graph"] = graph

        for key in (
                "intervention_target",
                "intervention_baseline",
                "intervention_mask",
                "action_utility_target",
                "replay_target"):
            raw_value = row.get(f"{key}_file") or row.get(key)
            if raw_value in (None, ""):
                continue
            path = Path(raw_value)
            path = path if path.is_absolute() else self.root / path
            if path.exists():
                out[key] = _load_array(path)
            elif key == "action_utility_target":
                try:
                    out[key] = torch.tensor(
                        float(raw_value), dtype=torch.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"sample {sample_id!r} has an invalid "
                        "action_utility_target") from exc
            else:
                raise FileNotFoundError(
                    f"sample {sample_id!r} contract file for {key!r} "
                    f"does not exist: {path}")

        for key in _STR_KEYS:
            if row.get(key):
                out[key] = row[key]
        sample_species = row.get("species") or self.species
        if sample_species:
            out["species"] = sample_species
        rate = row.get("rate_hz")
        out["dt"] = float(row.get("dt_s") or (
            (1.0 / float(rate)) if rate else
            1.0 / modality_info[0]["rate"]))
        out["window_start_s"] = float(valid_times[chosen_idx])
        raw_label = row.get(
            "cross_modal_label", row.get("cross_modal_labels"))
        if raw_label not in (None, ""):
            try:
                label = float(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sample {sample_id!r} has an invalid "
                    "cross_modal_label") from exc
            if label not in (0.0, 1.0):
                raise ValueError(
                    "cross_modal_label must be 0 (async) or 1 (synchronized)")
            out["cross_modal_labels"] = torch.tensor(
                label, dtype=torch.float32)
        return out

    def _align_to_union(self, window: torch.Tensor, mask: torch.Tensor,
                        ids: List[str], sample_id: str,
                        modality: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        union = self._global_ids
        position = {cid: i for i, cid in enumerate(union)}
        rows = [position[cid] for cid in ids if cid in position]
        if not rows:
            raise ValueError(f"sample {sample_id} has no channels in the union")
        C, T = window.shape
        aligned = torch.zeros(len(union), T, dtype=window.dtype)
        aligned_mask = torch.zeros(len(union), T, dtype=torch.bool)
        idx = torch.tensor(rows, dtype=torch.long)
        aligned[idx] = window[: len(rows)]
        aligned_mask[idx] = mask[: len(rows)]
        return aligned, aligned_mask, list(union)


# --------------------------------------------------------------------- splits
def split_rows(
    rows: Sequence[Dict[str, str]],
    by: str = "subject",
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]],
           Optional[List[Dict[str, str]]]]:
    """Group-preserving split (subjects/conditions/origins never split).

    ``by='none'`` falls back to plain row shuffling.
    """
    if by == "none":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows)).tolist()
        shuffled = [rows[i] for i in order]
    else:
        groups: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(row.get(by, "unknown"), []).append(row)
        keys = sorted(groups)
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)
        shuffled = [row for k in keys for row in groups[k]]
    n = len(shuffled)
    val_n = int(round(n * val_frac))
    if by != "none" and val_frac > 0:
        val_n = max(1, val_n)
    elif by == "none" and val_n == 0:
        val_n = 0  # caller asked for no validation split
    test_n = int(round(n * test_frac)) if test_frac else 0
    if by != "none":
        # group counts
        boundaries: List[int] = []
        current = 0
        for k in keys:
            g = len(groups[k])
            current += g
            boundaries.append(current)
        test_cut = next((b for b in boundaries if b >= n - test_n), n)
        val_cut = next((b for b in boundaries if b >= test_cut - val_n), test_cut)
    else:
        val_cut, test_cut = n - val_n - test_n, n - test_n
    train = shuffled[:val_cut]
    val = shuffled[val_cut:test_cut]
    test = shuffled[test_cut:] if test_n else None
    return train, val, test


# ------------------------------------------------------------------ collate
def species_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack equal tensors; pad 3-D tensors along time to the batch max.

    Keys absent from any sample are dropped from the batch (federation);
    strings stay as lists; ``graph`` (C, C) is padded to max C when shapes
    differ. Masks are padded with False (via their own zero padding since
    bool zero == False).
    """
    if not batch:
        return {}
    out: Dict[str, Any] = {}
    first = batch[0]
    for key in first:
        vals = [d.get(key) for d in batch]
        if any(v is None for v in vals):
            continue
        if isinstance(vals[0], torch.Tensor):
            if len({tuple(v.shape) for v in vals}) == 1:
                out[key] = torch.stack(vals, dim=0)
                continue
            shapes = [v.shape for v in vals]
            ndim = len(shapes[0])
            if key == "graph" and ndim == 2 and all(
                    s[0] == s[1] for s in shapes):
                # square adjacency over possibly differing channel counts
                max_c = max(s[0] for s in shapes)
                padded = []
                for v in vals:
                    if v.shape[0] < max_c:
                        padded.append(torch.nn.functional.pad(
                            v, (0, max_c - v.shape[0]) * 2))
                    else:
                        padded.append(v)
                out[key] = torch.stack(padded, dim=0)
                continue
            # Generic rule: pad the LAST axis when all other axes match
            # (per-sample windows differ in time length; masks follow their
            # modality and pad with False == zero).
            if any(s[:-1] != shapes[0][:-1] for s in shapes):
                raise ValueError(
                    f"cannot stack key {key} with shapes {shapes}; use "
                    f"align_channels/region_map for channel mismatch or "
                    f"batch-1")
            max_t = max(s[-1] for s in shapes)
            padded = []
            for v in vals:
                if v.shape[-1] < max_t:
                    padded.append(torch.nn.functional.pad(
                        v, (0, max_t - v.shape[-1])))
                else:
                    padded.append(v)
            out[key] = torch.stack(padded, dim=0)
        else:
            out[key] = list(vals)
    return out


# ---------------------------------------------------------------- builders
def _fit_global_normalization(
    root: Path,
    rows: Sequence[Dict[str, str]],
    modalities: Sequence[str],
    roles: Optional[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """Fit leakage-safe scalar statistics from training rows only."""
    role_map = dict(roles or {})
    signal_modalities = [
        modality for modality in modalities
        if role_map.get(modality, "signal") == "signal"
    ]
    if not signal_modalities or not rows:
        return {}
    probe = SpeciesSignalDataset(
        root, signal_modalities, seq_len=1, manifest=None,
        rows=rows, roles=role_map)
    stats: Dict[str, Dict[str, float]] = {}
    for modality in signal_modalities:
        total = 0
        total_sum = 0.0
        total_sq_sum = 0.0
        for row in rows:
            fpath = probe._row_file(row, modality)
            if fpath is None:
                continue
            data = _load_array(fpath)
            if data.dim() != 2:
                raise ValueError(f"{fpath} is not a (C, T) array")
            valid = torch.isfinite(data)
            mask_path = probe._row_file(row, modality, suffix_dir="mask")
            if mask_path is not None:
                mask = _load_array(mask_path)
                if mask.shape == data.shape:
                    valid = valid & mask.bool()
            values = data[valid].double()
            if values.numel() == 0:
                continue
            total += int(values.numel())
            total_sum += float(values.sum().item())
            total_sq_sum += float((values * values).sum().item())
        if total:
            mean = total_sum / total
            variance = max(total_sq_sum / total - mean * mean, 1e-12)
            stats[modality] = {
                "mean": mean,
                "std": float(np.sqrt(variance)),
            }
    return stats


def build_species_dataloaders(
    data_dir: Union[str, Path],
    modalities: Sequence[str],
    batch_size: int = 16,
    seq_len: Optional[int] = None,
    seq_seconds: Optional[float] = None,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    split_by: str = "subject",
    manifest: Union[str, Path, None] = "manifest.csv",
    load_ids: bool = False,
    align_channels: bool = False,
    max_union_channels: int = 4096,
    region_map: Union[str, Path, None] = None,
    load_graphs: bool = False,
    use_trials: bool = True,
    return_masks: bool = True,
    rate_default: Optional[float] = None,
    return_next_step_targets: bool = False,
    future_steps: int = 1,
    roles: Optional[Dict[str, str]] = None,
    species: Optional[str] = None,
    strict_rate: bool = True,
    expected_rate_hz: Optional[float] = None,
    normalization: Optional[str] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Union[Tuple[DataLoader, DataLoader],
           Tuple[DataLoader, DataLoader, DataLoader]]:
    """Train/val(/test) loaders with group-preserving splits.

    ``return_next_step_targets`` makes every sample expose
    ``<modality>_next`` for ``future_steps=1``. For ``future_steps > 1`` it
    additionally exposes ``<modality>_future`` with shape ``(K,C,T)`` and
    horizon-specific ``<modality>_next_<k>`` tensors. Validation and test
    datasets always use their first eligible window so metrics are repeatable;
    only the training dataset samples windows randomly.

    ``normalization="global_zscore"`` fits one global mean/std per signal
    modality on training rows only and reuses those statistics for validation
    and test rows; controls and auxiliary modalities remain unchanged.

    ``species``/``expected_rate_hz`` attach the species identity and the
    single-rate contract of the caller's profile. Each species trains its own
    model against its own ladder; the *code path* is shared, the model instance
    is not.
    """
    if seq_len is None and seq_seconds is None:
        seq_len = 256
    root = Path(data_dir)
    manifest_rows = read_manifest(
        root / manifest if manifest and not Path(manifest).is_absolute()
        else Path(manifest) if manifest else None)
    normalization_mode = str(normalization or "none").lower()
    if normalization_mode not in {"none", "global_zscore"}:
        raise ValueError(
            "normalization must be 'none' or 'global_zscore'")
    base = dict(
        data_dir=root, modalities=modalities,
        seq_len=seq_len, seq_seconds=seq_seconds, manifest=manifest,
        load_ids=load_ids, align_channels=align_channels,
        max_union_channels=max_union_channels, region_map=region_map,
        load_graphs=load_graphs, return_masks=return_masks,
        use_trials=use_trials, rate_default=rate_default,
        return_next_step_targets=return_next_step_targets, roles=roles,
        species=species, expected_rate_hz=expected_rate_hz,
        strict_rate=strict_rate, normalization=normalization_mode,
        future_steps=future_steps,
        normalization_stats={})

    if manifest_rows:
        rows = manifest_rows
    else:
        probe = SpeciesSignalDataset(
            root, modalities, seq_len=seq_len, seq_seconds=seq_seconds,
            manifest=None, rows=None, rate_default=rate_default,
            roles=roles, species=species,
            expected_rate_hz=expected_rate_hz, strict_rate=strict_rate)
        rows = [dict(r) for r in probe.rows]
    train_rows, val_rows, test_rows = split_rows(
        rows, by=split_by, seed=seed, val_frac=val_frac, test_frac=test_frac)
    if normalization_mode == "global_zscore":
        base["normalization_stats"] = _fit_global_normalization(
            root, train_rows, modalities, roles)

    def loader(rowset, do_shuffle):
        ds = SpeciesSignalDataset(
            **{**base, "rows": rowset, "rng_seed": seed,
               "random_windows": bool(do_shuffle)})
        drop = do_shuffle and len(rowset) >= batch_size
        loader_kwargs = dict(
            batch_size=batch_size, shuffle=do_shuffle,
            num_workers=num_workers, drop_last=drop,
            collate_fn=species_collate, pin_memory=pin_memory)
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
        return DataLoader(ds, **loader_kwargs)

    train_loader = loader(train_rows, shuffle)
    val_loader = loader(val_rows, False)
    if test_rows is not None:
        return train_loader, val_loader, loader(test_rows, False)
    return train_loader, val_loader
