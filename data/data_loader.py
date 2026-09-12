"""
Data loading pipeline for Brain MoE-PINN.

Supports:
- EEG datasets with arbitrary channel configurations (BIOT 3D positions)
- EEGdenoiseNet clean/artifact epochs for Magi EEG pretraining
- fMRI ROI time series (400 brain regions)
- Paired EEG-fMRI data for cross-modal training
- Standard 10-20 and 10-5 electrode systems
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import json


class EEGDenoiseNetDataset(Dataset):
    """Load EEGdenoiseNet epochs for Magi EEG pretraining.

    EEGdenoiseNet publishes clean EEG plus separate EOG/EMG artifact
    epochs.  This adapter keeps the source split deterministic and creates
    one noisy view with the repository's RMS/SNR convention.  The primary
    ``eeg`` view is clean EEG; ``eeg_view2`` is the matched noisy view.

    The source files are expected under ``root`` or ``root/data`` and may
    be NumPy ``.npy`` or MATLAB ``.mat`` files.  The public repository can
    contain annex pointer files instead of materialized arrays; those are
    rejected with an actionable error instead of being passed to NumPy.

    Args:
        root: EEGdenoiseNet checkout or extracted dataset directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        artifact: ``"EOG"`` or ``"EMG"``.
        train_fraction: Fraction of clean epochs assigned to train.
        val_fraction: Fraction assigned to validation after train.
        seed: Seed for deterministic source pairing and SNR sampling.
        snr_db_range: Inclusive uniform SNR range for train/validation.
    """

    _ARTIFACT_FILES = {
        "EEG": ("EEG_all_epochs", "EEG_all_epochs_512hz"),
        "EOG": ("EOG_all_epochs",),
        "EMG": ("EMG_all_epochs", "EMG_all_epochs_512hz"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        artifact: str = "EOG",
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        seed: int = 0,
        snr_db_range: Tuple[float, float] = (-7.0, 2.0),
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.split = str(split).lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.artifact = str(artifact).upper()
        if self.artifact not in {"EOG", "EMG"}:
            raise ValueError("artifact must be 'EOG' or 'EMG'")
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")
        if not 0.0 <= float(val_fraction) < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")
        if float(train_fraction) + float(val_fraction) >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1")
        low, high = (float(value) for value in snr_db_range)
        if low > high:
            raise ValueError("snr_db_range must be ordered low to high")
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.snr_db_range = (low, high)

        clean = self._load_source("EEG")
        artifact = self._load_source(self.artifact)
        target_length = int(artifact.shape[-1])
        if clean.shape[-1] != target_length:
            clean = self._resample_epochs(clean, target_length)
        if self.artifact == "EMG":
            count = max(int(clean.shape[0]), int(artifact.shape[0]))
        else:
            count = min(int(clean.shape[0]), int(artifact.shape[0]))
        if count < 3:
            raise ValueError(
                "EEGdenoiseNet requires at least three clean/artifact epochs "
                "after pairing")
        clean_indices = np.arange(count, dtype=np.int64) % clean.shape[0]
        self.clean = np.ascontiguousarray(
            clean[clean_indices], dtype=np.float32)
        self.artifact_epochs = np.ascontiguousarray(
            artifact, dtype=np.float32)
        self.sample_rate = 512 if self.artifact == "EMG" else 256
        indices = np.arange(count, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        train_end = min(
            count - 2, max(1, int(round(count * self.train_fraction))))
        val_end = min(
            count - 1,
            max(train_end + 1,
                train_end + int(round(count * self.val_fraction))),
        )
        if self.split == "train":
            self.indices = indices[:train_end]
        elif self.split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]
        if len(self.indices) == 0:
            raise ValueError(
                f"EEGdenoiseNet {self.split} split is empty; adjust split "
                "fractions")

    @staticmethod
    def _resample_epochs(epochs: np.ndarray, target_length: int) -> np.ndarray:
        """Linearly resample ``(N, T)`` epochs without a SciPy dependency."""
        source_length = int(epochs.shape[-1])
        source_grid = np.linspace(0.0, 1.0, source_length)
        target_grid = np.linspace(0.0, 1.0, int(target_length))
        return np.stack(
            [np.interp(target_grid, source_grid, epoch) for epoch in epochs],
            axis=0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _load_matrix(path: Path) -> np.ndarray:
        if path.stat().st_size < 1024:
            try:
                prefix = path.read_bytes()[:128]
            except OSError:
                prefix = b""
            if prefix.startswith(b"/annex/") or prefix.startswith(b"annex/"):
                raise RuntimeError(
                    f"EEGdenoiseNet file {path} is an annex pointer, not "
                    "materialized data; fetch the dataset object first")
        if path.suffix.lower() == ".npy":
            matrix = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".mat":
            try:
                from scipy.io import loadmat
            except ImportError as exc:
                raise RuntimeError(
                    "scipy is required to load EEGdenoiseNet .mat files; "
                    "use the published .npy files instead") from exc
            values = loadmat(path)
            candidates = [
                np.asarray(value)
                for key, value in values.items()
                if not key.startswith("__")
                and np.asarray(value).ndim >= 2
            ]
            if not candidates:
                raise ValueError(f"no matrix variable found in {path}")
            matrix = max(candidates, key=lambda value: value.size)
        else:
            raise ValueError(f"unsupported EEGdenoiseNet file type: {path}")
        matrix = np.asarray(matrix, dtype=np.float32).squeeze()
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2:
            raise ValueError(
                f"EEGdenoiseNet expects (epochs, samples), got "
                f"{matrix.shape} from {path}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"EEGdenoiseNet contains non-finite values: {path}")
        return matrix

    def _load_source(self, source: str) -> np.ndarray:
        names = self._ARTIFACT_FILES[source]
        if self.artifact == "EMG" and source in {"EEG", "EMG"}:
            names = tuple(reversed(names))
        candidates = []
        for name in names:
            for directory in (self.root, self.root / "data"):
                candidates.extend(
                    [directory / f"{name}.npy", directory / f"{name}.mat"])
        pointer_paths = []
        for path in candidates:
            if not path.is_file():
                continue
            try:
                return self._load_matrix(path)
            except RuntimeError as exc:
                if "annex pointer" in str(exc):
                    pointer_paths.append(path)
                    continue
                raise
        if pointer_paths:
            listed = ", ".join(str(path) for path in pointer_paths)
            raise RuntimeError(
                "EEGdenoiseNet arrays are present only as annex pointers: "
                f"{listed}. Materialize the dataset files before training.")
        expected = ", ".join(
            f"{name}.npy/.mat" for name in names)
        raise FileNotFoundError(
            f"cannot find EEGdenoiseNet {source} epochs below {self.root}; "
            f"expected {expected}")

    def _snr_db(self, item_index: int) -> float:
        if self.split == "test":
            levels = np.linspace(
                self.snr_db_range[0], self.snr_db_range[1], num=10)
            return float(levels[item_index % len(levels)])
        rng = np.random.default_rng(
            self.seed + 104729 * (int(self.indices[item_index]) + 1))
        return float(rng.uniform(*self.snr_db_range))

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item_index: int) -> Dict[str, object]:
        source_index = int(self.indices[item_index])
        clean = self.clean[source_index]
        noise_index = (source_index * 1009 + self.seed) % len(
            self.artifact_epochs)
        noise = self.artifact_epochs[noise_index]
        snr_db = self._snr_db(item_index)

        clean_rms = float(np.sqrt(np.mean(np.square(clean))))
        noise_rms = float(np.sqrt(np.mean(np.square(noise))))
        if clean_rms <= 1e-8 or noise_rms <= 1e-8:
            raise ValueError(
                "EEGdenoiseNet epoch has near-zero RMS; cannot synthesize "
                "a stable noisy view")
        # Match the benchmark code's dB-to-amplitude convention.
        scale = clean_rms / (noise_rms * (10.0 ** (0.1 * snr_db)))
        noisy = clean + scale * noise
        noisy_std = float(np.std(noisy))
        if noisy_std <= 1e-8:
            raise ValueError(
                "EEGdenoiseNet synthesized epoch has near-zero standard "
                "deviation")
        clean = clean / noisy_std
        noisy = noisy / noisy_std
        clean_tensor = torch.from_numpy(
            np.ascontiguousarray(clean[None, :], dtype=np.float32))
        noisy_tensor = torch.from_numpy(
            np.ascontiguousarray(noisy[None, :], dtype=np.float32))
        return {
            "eeg": clean_tensor,
            "eeg_view1": clean_tensor.clone(),
            "eeg_view2": noisy_tensor,
            "eeg_noisy": noisy_tensor,
            "eeg_target": clean_tensor.clone(),
            "channel_names": ["EEGdenoiseNet"],
            "channel_types": torch.zeros(1, dtype=torch.long),
            "sample_rate_hz": self.sample_rate,
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "artifact_type": self.artifact,
            "sample_id": f"{self.split}-{source_index}",
        }

    @staticmethod
    def collate_fn(samples: List[Dict[str, object]]) -> Dict[str, object]:
        """Collate tensors while preserving per-sample channel metadata."""
        if not samples:
            raise ValueError("cannot collate an empty EEGdenoiseNet batch")
        result: Dict[str, object] = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "channel_names":
                result[key] = values
            elif key == "artifact_type" or key == "sample_id":
                result[key] = values
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.as_tensor(values)
        return result


class EEGDataset(Dataset):
    """
    EEG dataset with support for arbitrary montages and BIOT 3D positions.

    Args:
        data_dir: Directory containing EEG recordings
        sample_rate: Target sample rate (Hz)
        seq_duration: Sequence duration in seconds
        patch_size: Temporal patch size (samples)
        stride: Patch stride (samples)
        channel_names: Optional list of channel names for BIOT embedding
        transform: Optional transform to apply
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 256,
        seq_duration: float = 10.0,
        patch_size: int = 256,
        stride: int = 128,
        channel_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        preload: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.patch_size = patch_size
        self.stride = stride
        self.channel_names = channel_names or ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3", "C4", "T4",
                                               "T5", "P3", "P4", "T6", "O1", "O2", "Fz", "Cz", "Pz"]
        self.num_channels = len(self.channel_names)
        self.transform = transform
        self.preload = preload

        self.seq_length = int(seq_duration * sample_rate)
        self.num_patches = (self.seq_length - patch_size) // stride + 1

        self.file_list = []
        self._scan_data_dir()

        if self.preload:
            self.data_cache = []
            self._preload_data()

    def _scan_data_dir(self):
        """Scan data directory for EEG files."""
        supported_formats = [".npy", ".npz", ".pth", ".edf", ".set"]
        for ext in supported_formats:
            self.file_list.extend(list(self.data_dir.glob(f"**/*{ext}")))
        self.file_list = [str(f) for f in self.file_list if f.is_file()]
        print(f"[EEGDataset] Found {len(self.file_list)} EEG files in {self.data_dir}")

    def _preload_data(self):
        """Preload data into memory."""
        print(f"[EEGDataset] Preloading {len(self.file_list)} files...")
        for fpath in self.file_list:
            try:
                if fpath.endswith(".npy"):
                    data = np.load(fpath)
                elif fpath.endswith(".npz"):
                    data = np.load(fpath)["data"]
                elif fpath.endswith(".pth"):
                    data = torch.load(fpath, weights_only=True).numpy()
                else:
                    continue
                self.data_cache.append(data)
            except Exception as e:
                print(f"[EEGDataset] Warning: Failed to load {fpath}: {e}")
        print(f"[EEGDataset] Preloaded {len(self.data_cache)} files")

    def __len__(self) -> int:
        return len(self.file_list) if not self.preload else len(self.data_cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            eeg_data = self.data_cache[idx]
        else:
            fpath = self.file_list[idx]
            if fpath.endswith(".npy"):
                eeg_data = np.load(fpath)
            elif fpath.endswith(".npz"):
                eeg_data = np.load(fpath)["data"]
            elif fpath.endswith(".pth"):
                eeg_data = torch.load(fpath).numpy()
            else:
                raise ValueError(f"Unsupported file format: {fpath}")

        if isinstance(eeg_data, np.ndarray):
            eeg_data = torch.from_numpy(eeg_data).float()

        if eeg_data.shape[0] != self.num_channels:
            if eeg_data.shape[1] == self.num_channels:
                eeg_data = eeg_data.T
            else:
                target_len = self.num_channels * self.seq_length
                if eeg_data.numel() >= target_len:
                    eeg_data = eeg_data[:self.num_channels, :self.seq_length]
                else:
                    pad_len = target_len - eeg_data.numel()
                    eeg_data = torch.nn.functional.pad(eeg_data.flatten(), (0, pad_len))
                    eeg_data = eeg_data.reshape(self.num_channels, -1)[:, :self.seq_length]

        if eeg_data.shape[1] > self.seq_length:
            start_idx = torch.randint(0, eeg_data.shape[1] - self.seq_length, (1,)).item()
            eeg_data = eeg_data[:, start_idx:start_idx + self.seq_length]
        elif eeg_data.shape[1] < self.seq_length:
            eeg_data = torch.nn.functional.pad(eeg_data, (0, self.seq_length - eeg_data.shape[1]))

        if self.transform:
            eeg_data = self.transform(eeg_data)

        return {
            "eeg": eeg_data,
            "channel_names": self.channel_names,
            "file_idx": idx,
        }


class fMRIDataset(Dataset):
    """
    fMRI ROI time series dataset.

    Args:
        data_dir: Directory containing fMRI recordings
        num_regions: Number of brain regions (default 400 for ROI parcellation)
        seq_length: Sequence length in TRs
        tr: Repetition time in seconds (default 2.0s for BOLD)
        transform: Optional transform to apply
    """

    def __init__(
        self,
        data_dir: str,
        num_regions: int = 400,
        seq_length: int = 100,
        tr: float = 2.0,
        transform: Optional[Callable] = None,
        preload: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.num_regions = num_regions
        self.seq_length = seq_length
        self.tr = tr
        self.transform = transform
        self.preload = preload

        self.file_list = []
        self._scan_data_dir()

        if self.preload:
            self.data_cache = []
            self._preload_data()

    def _scan_data_dir(self):
        """Scan data directory for fMRI files."""
        supported_formats = [".npy", ".npz", ".pth", ".csv"]
        for ext in supported_formats:
            self.file_list.extend(list(self.data_dir.glob(f"**/*{ext}")))
        self.file_list = [str(f) for f in self.file_list if f.is_file()]
        print(f"[fMRIDataset] Found {len(self.file_list)} fMRI files in {self.data_dir}")

    def _preload_data(self):
        """Preload data into memory."""
        print(f"[fMRIDataset] Preloading {len(self.file_list)} files...")
        for fpath in self.file_list:
            try:
                if fpath.endswith(".npy"):
                    data = np.load(fpath)
                elif fpath.endswith(".npz"):
                    data = np.load(fpath)["data"]
                elif fpath.endswith(".pth"):
                    data = torch.load(fpath, weights_only=True).numpy()
                elif fpath.endswith(".csv"):
                    data = np.loadtxt(fpath, delimiter=",", skiprows=1)
                else:
                    continue
                self.data_cache.append(data)
            except Exception as e:
                print(f"[fMRIDataset] Warning: Failed to load {fpath}: {e}")
        print(f"[fMRIDataset] Preloaded {len(self.data_cache)} files")

    def __len__(self) -> int:
        return len(self.file_list) if not self.preload else len(self.data_cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            fmri_data = self.data_cache[idx]
        else:
            fpath = self.file_list[idx]
            if fpath.endswith(".npy"):
                fmri_data = np.load(fpath)
            elif fpath.endswith(".npz"):
                fmri_data = np.load(fpath)["data"]
            elif fpath.endswith(".pth"):
                fmri_data = torch.load(fpath).numpy()
            elif fpath.endswith(".csv"):
                fmri_data = np.loadtxt(fpath, delimiter=",", skiprows=1)
            else:
                raise ValueError(f"Unsupported file format: {fpath}")

        if isinstance(fmri_data, np.ndarray):
            fmri_data = torch.from_numpy(fmri_data).float()

        if fmri_data.shape[0] != self.num_regions:
            if fmri_data.shape[1] == self.num_regions:
                fmri_data = fmri_data.T

        if fmri_data.shape[1] > self.seq_length:
            start_idx = torch.randint(0, fmri_data.shape[1] - self.seq_length, (1,)).item()
            fmri_data = fmri_data[:, start_idx:start_idx + self.seq_length]
        elif fmri_data.shape[1] < self.seq_length:
            fmri_data = torch.nn.functional.pad(fmri_data, (0, self.seq_length - fmri_data.shape[1]))

        if self.transform:
            fmri_data = self.transform(fmri_data)

        return {
            "fmri": fmri_data,
            "num_regions": self.num_regions,
            "tr": self.tr,
            "file_idx": idx,
        }


class MEGDataset(Dataset):
    """
    MEG dataset for 306-channel Neuromag systems.

    Args:
        data_dir: Directory containing MEG recordings (.npy, .npz, .fif)
        num_channels: Number of MEG sensors (default 306 for Elekta Neuromag)
        sample_rate: Target sample rate (Hz)
        seq_duration: Sequence duration in seconds
        preload: Whether to load all data into memory
    """

    def __init__(
        self,
        data_dir: str,
        num_channels: int = 306,
        sample_rate: int = 1000,
        seq_duration: float = 10.0,
        transform: Optional[Callable] = None,
        preload: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.seq_length = int(seq_duration * sample_rate)
        self.transform = transform
        self.preload = preload

        self.file_list = []
        self._scan_data_dir()

        if self.preload:
            self.data_cache = []
            self._preload_data()

    def _scan_data_dir(self):
        supported_formats = [".npy", ".npz", ".pth"]
        for ext in supported_formats:
            self.file_list.extend(list(self.data_dir.glob(f"**/*{ext}")))
        self.file_list = [str(f) for f in self.file_list if f.is_file()]
        print(f"[MEGDataset] Found {len(self.file_list)} MEG files in {self.data_dir}")

    def _preload_data(self):
        print(f"[MEGDataset] Preloading {len(self.file_list)} files...")
        for fpath in self.file_list:
            try:
                if fpath.endswith(".npy"):
                    data = np.load(fpath)
                elif fpath.endswith(".npz"):
                    data = np.load(fpath)["data"]
                elif fpath.endswith(".pth"):
                    data = torch.load(fpath, weights_only=True).numpy()
                else:
                    continue
                self.data_cache.append(data)
            except Exception as e:
                print(f"[MEGDataset] Warning: Failed to load {fpath}: {e}")
        print(f"[MEGDataset] Preloaded {len(self.data_cache)} files")

    def __len__(self) -> int:
        return len(self.file_list) if not self.preload else len(self.data_cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            meg_data = self.data_cache[idx]
        else:
            fpath = self.file_list[idx]
            if fpath.endswith(".npy"):
                meg_data = np.load(fpath)
            elif fpath.endswith(".npz"):
                meg_data = np.load(fpath)["data"]
            elif fpath.endswith(".pth"):
                meg_data = torch.load(fpath).numpy()
            else:
                raise ValueError(f"Unsupported MEG file format: {fpath}")

        meg_tensor = torch.FloatTensor(meg_data)

        if self.transform:
            meg_tensor = self.transform(meg_tensor)

        return {
            "meg": meg_tensor,
            "num_channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "file_idx": idx,
        }


class PairedBrainDataset(Dataset):
    """
    Paired EEG-fMRI (optionally MEG) dataset for cross-modal training.

    Expects directory structure (aligned by filename stem):
        data_dir/
            eeg/
                subject1_run1.npy
                subject1_run2.npy
            fmri/
                subject1_run1.npy
                subject1_run2.npy
            meg/            (optional; stems shared with eeg/fmri)
                subject1_run1.npy
            metadata.json  (alignment info, optional)
    """

    def __init__(
        self,
        data_dir: str,
        eeg_dataset: Optional[EEGDataset] = None,
        fmri_dataset: Optional[fMRIDataset] = None,
        alignment_file: Optional[str] = None,
        return_next_step_targets: bool = False,
        future_steps: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.eeg_dataset = eeg_dataset
        self.fmri_dataset = fmri_dataset
        self.return_next_step_targets = bool(return_next_step_targets)
        self.future_steps = int(future_steps)
        if self.future_steps < 1:
            raise ValueError("future_steps must be positive")
        self.pair_list = []
        self.metadata = {}

        if alignment_file:
            self._load_alignment(alignment_file)
        else:
            self._auto_align()

    def _load_alignment(self, alignment_file: str):
        """Load path mappings from an alignment JSON file."""
        with open(alignment_file, "r") as f:
            raw = json.load(f)
        self.metadata = raw
        if not isinstance(raw, dict):
            raise ValueError("alignment JSON must map sample IDs to records")
        for sample_id, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"alignment record {sample_id!r} must be an object")
            entry = dict(value)
            entry.setdefault("sample_id", sample_id)
            self.pair_list.append(entry)
        print(f"[PairedBrainDataset] Loaded {len(self.pair_list)} aligned pairs from {alignment_file}")

    def _auto_align(self):
        """Automatically align EEG and fMRI (and optional MEG) by filename.

        Expected layout: ``data_dir/{eeg,fmri,meg}/<stem>.npy`` — a sample is
        a (C, T) or (T, C)-read raw array; MEG is optional per stem.
        """
        eeg_dir = self.data_dir / "eeg"
        fmri_dir = self.data_dir / "fmri"
        meg_dir = self.data_dir / "meg"

        if not eeg_dir.exists() or not fmri_dir.exists():
            print("[PairedBrainDataset] Warning: No aligned data found, using unaligned mode")
            return

        def _scan(directory):
            if not directory.exists():
                return {}
            return {f.stem: str(f) for f in directory.glob("*")
                    if f.is_file() and f.suffix in (".npy", ".npz", ".pth", ".csv")}

        eeg_files = _scan(eeg_dir)
        fmri_files = _scan(fmri_dir)
        meg_files = _scan(meg_dir)

        common_keys = set(eeg_files.keys()) & set(fmri_files.keys())
        for key in sorted(common_keys):
            entry = {"eeg": eeg_files[key], "fmri": fmri_files[key]}
            if key in meg_files:
                entry["meg"] = meg_files[key]
            self.pair_list.append(entry)
            self.metadata[key] = entry

        print(f"[PairedBrainDataset] Auto-aligned {len(self.pair_list)} pairs "
              f"({sum('meg' in e for e in self.pair_list)} with MEG)")

    def __len__(self) -> int:
        return len(self.pair_list)

    @staticmethod
    def _load_pair_array(fpath: str, data_dir: Path) -> torch.Tensor:
        path = Path(fpath)
        if not path.is_absolute():
            path = data_dir / path
        if path.suffix == ".npy":
            data = np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)["data"]
        elif path.suffix == ".pth":
            data = torch.load(path, weights_only=True)
            data = data.numpy() if hasattr(data, "numpy") else data
        elif path.suffix == ".csv":
            data = np.loadtxt(path, delimiter=",")
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return torch.as_tensor(data).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.pair_list:
            empty = {"is_paired": False}
            if self.eeg_dataset is not None:
                empty["eeg"] = torch.zeros(
                    self.eeg_dataset.num_channels,
                    self.eeg_dataset.seq_length)
            if self.fmri_dataset is not None:
                empty["fmri"] = torch.zeros(
                    self.fmri_dataset.num_regions,
                    self.fmri_dataset.seq_length)
            return empty

        entry = self.pair_list[idx]
        out = {}
        for modality in ("eeg", "fmri", "meg"):
            fpath = entry.get(modality)
            if not fpath:
                continue
            data = self._load_pair_array(fpath, self.data_dir)
            if not self.return_next_step_targets:
                out[modality] = data
                continue

            futures = []
            explicit = True
            for horizon in range(1, self.future_steps + 1):
                future_path = entry.get(f"{modality}_next_{horizon}")
                if horizon == 1 and not future_path:
                    future_path = entry.get(f"{modality}_next")
                if not future_path:
                    explicit = False
                    break
                futures.append(self._load_pair_array(
                    future_path, self.data_dir))

            if not explicit:
                packed_path = entry.get(f"{modality}_future")
                if packed_path:
                    packed = self._load_pair_array(
                        packed_path, self.data_dir)
                    if (packed.ndim == data.ndim + 1
                            and packed.shape[0] == self.future_steps):
                        futures = [packed[h] for h in range(self.future_steps)]
                    elif packed.shape[-1] % self.future_steps == 0:
                        future_len = packed.shape[-1] // self.future_steps
                        futures = [
                            packed[..., h * future_len:(h + 1) * future_len]
                            for h in range(self.future_steps)]
                    else:
                        raise ValueError(
                            f"sample {entry.get('sample_id', idx)!r}, modality "
                            f"{modality!r} future array has no {self.future_steps}"
                            " equal horizons")
                else:
                    segments = self.future_steps + 1
                    if data.ndim == 0 or data.shape[-1] % segments:
                        length_requirement = (
                            "even-length" if segments == 2
                            else f"length divisible by {segments}")
                        raise ValueError(
                            f"sample {entry.get('sample_id', idx)!r}, modality "
                            f"{modality!r} needs a {length_requirement} signal "
                            f"split into {segments} equal context/future "
                            "segments or explicit future paths")
                    context_len = data.shape[-1] // segments
                    futures = [
                        data[..., (h + 1) * context_len:
                             (h + 2) * context_len]
                        for h in range(self.future_steps)]
                    data = data[..., :context_len]

            if any(future.shape != data.shape for future in futures):
                raise ValueError(
                    f"sample {entry.get('sample_id', idx)!r}, modality "
                    f"{modality!r} context and future shapes must match")
            out[modality] = data
            if self.future_steps == 1:
                out[f"{modality}_next"] = futures[0]
            else:
                out[f"{modality}_future"] = torch.stack(futures, dim=0)
                for horizon, future in enumerate(futures, start=1):
                    out[f"{modality}_next_{horizon}"] = future
        for key in (
                "intervention_target",
                "intervention_baseline",
                "intervention_mask",
                "action_utility_target",
                "replay_target"):
            value = entry.get(key)
            if value is None:
                continue
            if isinstance(value, (str, Path)):
                value = self._load_pair_array(value, self.data_dir)
            else:
                value = torch.as_tensor(value).float()
            out[key] = value
        label = entry.get(
            "cross_modal_label", entry.get("cross_modal_labels", 1.0))
        try:
            label = float(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pair {entry.get('sample_id', idx)!r} has an invalid "
                "cross_modal_label") from exc
        if label not in (0.0, 1.0):
            raise ValueError(
                "cross_modal_label must be 0 (async) or 1 (synchronized)")
        out["cross_modal_labels"] = torch.tensor(label, dtype=torch.float32)
        out["is_paired"] = True
        if entry.get("sample_id"):
            out["sample_id"] = entry["sample_id"]
        return out


def create_brain_dataloaders(
    eeg_data_dir: str,
    fmri_data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    eeg_sample_rate: int = 256,
    fmri_seq_length: int = 100,
    eeg_seq_duration: float = 10.0,
    patch_size: int = 256,
    stride: int = 128,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create EEG and fMRI dataloaders.

    Args:
        eeg_data_dir: Path to EEG data directory
        fmri_data_dir: Path to fMRI data directory
        batch_size: Batch size
        num_workers: Number of dataloader workers
        eeg_sample_rate: EEG sample rate (Hz)
        fmri_seq_length: fMRI sequence length (TRs)
        eeg_seq_duration: EEG sequence duration (seconds)
        patch_size: EEG patch size
        stride: EEG patch stride
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        eeg_loader, fmri_loader
    """
    eeg_dataset = EEGDataset(
        data_dir=eeg_data_dir,
        sample_rate=eeg_sample_rate,
        seq_duration=eeg_seq_duration,
        patch_size=patch_size,
        stride=stride,
        preload=False,
    )

    fmri_dataset = fMRIDataset(
        data_dir=fmri_data_dir,
        seq_length=fmri_seq_length,
        preload=False,
    )

    eeg_loader = DataLoader(
        eeg_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    fmri_loader = DataLoader(
        fmri_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return eeg_loader, fmri_loader


class DataMixer:
    """
    Progress-dependent data mixing schedule.

    Controls the sampling ratio between resting-state, task-simple,
    and task-complex data based on training progress.

    Usage:
        mixer = DataMixer(rest_loader, task_simple_loader, task_complex_loader)
        batch = mixer.next_batch(step=current_step, total_steps=100000)
    """

    # Default schedule: (start_progress, end_progress, resting_ratio, task_simple_ratio, task_complex_ratio)
    DEFAULT_SCHEDULE = [
        (0.0, 0.1, 1.0, 0.0, 0.0),      # Phase 0: resting only
        (0.1, 0.3, 0.8, 0.2, 0.0),      # Phase 1: + simple tasks
        (0.3, 0.6, 0.6, 0.3, 0.1),      # Phase 2: + complex tasks
        (0.6, 0.8, 0.5, 0.3, 0.2),      # Phase 3: heavier tasks
        (0.8, 1.0, 0.4, 0.3, 0.3),      # Phase 4: balanced
    ]

    def __init__(
        self,
        resting_loader: DataLoader,
        task_simple_loader: Optional[DataLoader] = None,
        task_complex_loader: Optional[DataLoader] = None,
        schedule: Optional[List[Tuple[float, float, float, float, float]]] = None,
    ):
        self.resting_loader = resting_loader
        self.task_simple_loader = task_simple_loader
        self.task_complex_loader = task_complex_loader
        self.schedule = schedule or self.DEFAULT_SCHEDULE

        self._rest_iter = iter(resting_loader)
        self._simple_iter = task_simple_loader and iter(task_simple_loader)
        self._complex_iter = task_complex_loader and iter(task_complex_loader)

    def _get_progress(self, step: int, total_steps: int) -> float:
        return min(step / max(total_steps, 1), 1.0)

    def _get_ratios(self, progress: float) -> Tuple[float, float, float]:
        for start, end, r_rest, r_simple, r_complex in self.schedule:
            if start <= progress <= end:
                return r_rest, r_simple, r_complex
        # Fallback: use last schedule entry
        return 0.4, 0.3, 0.3

    def _next_from_loader(self, loader_iter):
        try:
            return next(loader_iter)
        except StopIteration:
            return None

    def next_batch(self, step: int, total_steps: int) -> Dict:
        """Get next batch with progress-dependent mixing."""
        progress = self._get_progress(step, total_steps)
        r_rest, r_simple, r_complex = self._get_ratios(progress)

        # Normalized probabilities
        total = r_rest + r_simple + r_complex
        p_rest = r_rest / total
        p_simple = r_simple / total
        p_complex = r_complex / total

        # Decide which loader to sample from
        rand = np.random.random()

        if rand < p_complex and self.task_complex_loader is not None and self._complex_iter is not None:
            batch = next(self._complex_iter, None)
            if batch is None:
                self._complex_iter = iter(self.task_complex_loader)
                batch = next(self._complex_iter)
            return batch
        elif rand < p_complex + p_simple and self.task_simple_loader is not None and self._simple_iter is not None:
            batch = next(self._simple_iter, None)
            if batch is None:
                self._simple_iter = iter(self.task_simple_loader)
                batch = next(self._simple_iter)
            return batch
        else:
            batch = next(self._rest_iter, None)
            if batch is None:
                self._rest_iter = iter(self.resting_loader)
                batch = next(self._rest_iter)
            return batch


if __name__ == "__main__":
    print("Testing data loading pipeline...")

    print("\nNote: Create dummy data directories to test actual loading")
    print("EEGDataset and fMRIDataset are ready for use with real data paths")
    print("Use create_brain_dataloaders() for standard training setup")