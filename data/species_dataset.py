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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_FORMATS = (".npy", ".npz", ".pth")
_STR_KEYS = ("sample_id", "subject", "session", "origin", "condition")
_META_KEYS = _STR_KEYS + ("dt",)


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
        rng_seed: int = 0,
        random_windows: bool = True,
        return_next_step_targets: bool = False,
        roles: Optional[Dict[str, str]] = None,
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
        self.rows: List[Dict[str, str]] = (
            list(rows) if rows is not None else manifest_rows)
        if not self.rows:
            self.rows = self._infer_rows()

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
            base = self.root / f"{modality}_{suffix_dir}"
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
        """Return the modality-specific sampling rate for one manifest row."""
        value = row.get(f"{modality}_rate_hz") or row.get(
            f"{modality}_hz")
        if value:
            rate = float(value)
        else:
            dt = row.get(f"{modality}_dt_s")
            rate = 1.0 / float(dt) if dt else float(
                row.get("rate_hz") or self.rate_default or 1.0)
        if rate <= 0:
            raise ValueError(
                f"sampling rate for modality {modality!r} must be positive")
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
            required = win * 2 if self.return_next_step_targets else win
            starts = self._allowed_starts(data.shape[1], trials, required)
            if not starts:
                if self.return_next_step_targets:
                    raise ValueError(
                        f"sample {sample_id!r}, modality {modality!r} has "
                        f"no context+future window of {win} frames")
                # Preserve padded-window behavior for observation-only use.
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
            # no modality present for this row: return empty metadata only
            for k in _STR_KEYS:
                if row.get(k):
                    out[k] = row[k]
            return out

        if self.seq_len is not None and len({
            info["rate"] for info in modality_info
        }) > 1:
            raise ValueError(
                "seq_len is frame-based and cannot align modalities with "
                "different sampling rates; use seq_seconds")

        # Candidate starts are represented in seconds.  Each modality is then
        # sampled on its own grid, preventing raw frame indices from being
        # mistaken for synchronized physical time.
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
            if mask_path is not None:
                mask_data = _load_array(mask_path)
                if mask_data.shape == data.shape:
                    mask_window, mask_valid = self._slice_window(
                        mask_data, start, win)
                    valid = valid & mask_valid & mask_window.bool()
            mask = valid

            next_window = next_mask = None
            if self.return_next_step_targets:
                next_window, next_valid = self._slice_window(
                    data, start + win, win)
                if mask_path is not None and mask_data.shape == data.shape:
                    next_mask_window, next_mask_valid = self._slice_window(
                        mask_data, start + win, win)
                    next_valid = (
                        next_valid & next_mask_valid
                        & next_mask_window.bool())
                next_mask = next_valid

            ids = self._sample_ids(row, modality) if (
                self.load_ids or self.align_channels or
                self.regions is not None) else None
            if ids is not None and len(ids) != window.shape[0]:
                ids = None  # channel count mismatch: treat as unlabeled
            if self.regions is not None and ids is not None:
                window, mask = _aggregate_regions(
                    window, mask, ids, self._regions, self.regions)
                if next_window is not None:
                    next_window, next_mask = _aggregate_regions(
                        next_window, next_mask, ids,
                        self._regions, self.regions)
                ids = self._regions
            if self.align_channels and ids is not None:
                source_ids = list(ids)
                window, mask, ids = self._align_to_union(
                    window, mask, source_ids, sample_id, modality)
                if next_window is not None:
                    next_window, next_mask, _ = self._align_to_union(
                        next_window, next_mask, source_ids,
                        sample_id, modality)
            out[modality] = window
            if self.return_masks:
                out[f"{modality}_mask"] = mask
            if self.load_ids:
                out[f"{modality}_ids"] = ids or []
            if next_window is not None:
                out[f"{modality}_next"] = next_window
                if self.return_masks:
                    out[f"{modality}_next_mask"] = next_mask

        if self.load_graphs:
            gpath = self._row_file(row, "graph", suffix_dir="graphs")
            gpath = gpath if gpath is not None else (
                self.root / "graphs" / f"{sample_id}.npy")
            if gpath and gpath.is_file():
                out["graph"] = _load_array(gpath)
        for k in _STR_KEYS:
            if row.get(k):
                out[k] = row[k]
        rate = row.get("rate_hz")
        out["dt"] = float(row.get("dt_s") or (
            (1.0 / float(rate)) if rate else
            1.0 / modality_info[0]["rate"]))
        out["window_start_s"] = float(valid_times[chosen_idx])
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
    return_masks: bool = True,
    use_trials: bool = True,
    rate_default: Optional[float] = None,
    return_next_step_targets: bool = False,
    roles: Optional[Dict[str, str]] = None,
) -> Union[Tuple[DataLoader, DataLoader],
           Tuple[DataLoader, DataLoader, DataLoader]]:
    """Train/val(/test) loaders with group-preserving splits.

    ``return_next_step_targets`` makes every sample expose
    ``<modality>_next`` from the adjacent non-overlapping window.  Validation
    and test datasets always use their first eligible window so metrics are
    repeatable; only the training dataset samples windows randomly.
    """
    if seq_len is None and seq_seconds is None:
        seq_len = 256
    root = Path(data_dir)
    manifest_rows = read_manifest(
        root / manifest if manifest and not Path(manifest).is_absolute()
        else Path(manifest) if manifest else None)
    base = dict(
        data_dir=root, modalities=modalities,
        seq_len=seq_len, seq_seconds=seq_seconds, manifest=manifest,
        load_ids=load_ids, align_channels=align_channels,
        max_union_channels=max_union_channels, region_map=region_map,
        load_graphs=load_graphs, return_masks=return_masks,
        use_trials=use_trials, rate_default=rate_default,
        return_next_step_targets=return_next_step_targets, roles=roles)

    if manifest_rows:
        rows = manifest_rows
    else:
        probe = SpeciesSignalDataset(
            root, modalities, seq_len=seq_len, seq_seconds=seq_seconds,
            manifest=None, rows=None, rate_default=rate_default,
            roles=roles)
        rows = [dict(r) for r in probe.rows]
    train_rows, val_rows, test_rows = split_rows(
        rows, by=split_by, seed=seed, val_frac=val_frac, test_frac=test_frac)

    def loader(rowset, do_shuffle):
        ds = SpeciesSignalDataset(
            **{**base, "rows": rowset, "rng_seed": seed,
               "random_windows": bool(do_shuffle)})
        drop = do_shuffle and len(rowset) >= batch_size
        return DataLoader(
            ds, batch_size=batch_size, shuffle=do_shuffle,
            num_workers=num_workers, drop_last=drop,
            collate_fn=species_collate)

    train_loader = loader(train_rows, shuffle)
    val_loader = loader(val_rows, False)
    if test_rows is not None:
        return train_loader, val_loader, loader(test_rows, False)
    return train_loader, val_loader
