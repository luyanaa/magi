"""
Data loading pipeline for Brain MoE-PINN.

Supports:
- EEG datasets with arbitrary channel configurations (BIOT 3D positions)
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
    ):
        self.data_dir = Path(data_dir)
        self.eeg_dataset = eeg_dataset
        self.fmri_dataset = fmri_dataset
        self.return_next_step_targets = bool(return_next_step_targets)

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

            next_path = entry.get(f"{modality}_next")
            if next_path:
                next_data = self._load_pair_array(next_path, self.data_dir)
            else:
                if data.ndim == 0 or data.shape[-1] < 2:
                    raise ValueError(
                        f"sample {entry.get('sample_id', idx)!r}, modality "
                        f"{modality!r} has no future target")
                if data.shape[-1] % 2:
                    raise ValueError(
                        f"sample {entry.get('sample_id', idx)!r}, modality "
                        f"{modality!r} needs an even-length context+future "
                        "array or an explicit *_next path")
                context_len = data.shape[-1] // 2
                next_data = data[..., context_len:]
                data = data[..., :context_len]
            out[modality] = data
            out[f"{modality}_next"] = next_data
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