"""
ECoG Dataset for Brain MoE-PINN.

Supports:
- DANDI NWB files (AJILE12, iEEG+fMRI sync, etc.)
- EDF files with ECoG/sEEG data
- BIDS-iEEG format
- Variable channel counts (cap 256)
- MNI coordinate extraction for BIOT embedding
- Channel type embedding (ECoG grid vs sEEG depth)
- Amplitude normalization (ECoG ~20× scalp EEG)
- Integration with existing EEGDataset pipeline

References:
- DANDI: https://dandiarchive.org
- AJILE12: DANDI:000055 (845 GB, 12 subjects, multi-day continuous)
- iEEG+fMRI sync: DANDI:000623
- Scalp+iEEG: DANDI:000574
- PtNRGrids: DANDI:000465/000554
- BIDS-iEEG: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/04-intracranial-electroencephalography.html
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Union, Any
from pathlib import Path
import json
import warnings

# Try to import MNE and related libraries
try:
    import mne
    from mne.io import read_raw_edf, read_raw_bdf, read_raw_fif, read_raw_eeglab
    from mne_bids import BIDSPath, read_raw_bids
    from pynwb import NWBHDF5IO
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE not available. ECoG dataset will use simplified loading.")

try:
    from dandi.dandiapi import DandiAPIClient
    DANDI_AVAILABLE = True
except ImportError:
    DANDI_AVAILABLE = False
    warnings.warn("DANDI client not available. Remote DANDI datasets cannot be downloaded.")

from .ecog_preprocessing import (
    preprocess_ecog,
    load_dandi_nwb,
    load_edf_ecog,
    extract_mni_coords,
    map_channel_type,
    CHANNEL_TYPES,
)


class ECoGDataset(Dataset):
    """
    ECoG/sEEG dataset with support for DANDI, BIDS, and raw formats.
    
    Handles:
    - Variable channel counts (1-256)
    - Mixed ECoG grid and sEEG depth electrodes
    - MNI coordinate extraction for BIOT embedding
    - Amplitude normalization (ECoG ~20× scalp EEG)
    - Channel type embedding
    - Integration with Magi v2 encoder
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 512,  # Higher for ECoG (70-200Hz HFB)
        seq_duration: float = 5.0,  # Shorter for ECoG (denser signals)
        patch_size: int = 256,
        stride: int = 128,
        max_channels: int = 256,
        ecog_amplitude_scale: float = 20.0,
        preload: bool = False,
        use_mni_coords: bool = True,
        require_mni: bool = False,
        data_format: str = "auto",  # "auto", "dandi", "bids", "edf", "nwb"
        dandi_id: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        runs: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize ECoG dataset.
        
        Args:
            data_dir: Directory containing ECoG data
            sample_rate: Target sample rate (Hz)
            seq_duration: Sequence duration in seconds
            patch_size: Temporal patch size (samples)
            stride: Patch stride (samples)
            max_channels: Maximum number of channels to include (cap for memory)
            ecog_amplitude_scale: Scaling factor for ECoG amplitude (20.0 = 1/20 scaling)
            preload: Whether to preload all data into memory
            use_mni_coords: Whether to extract MNI coordinates for BIOT embedding
            require_mni: Whether to require MNI coordinates (skip files without)
            data_format: Data format ("auto" detects from file extensions)
            dandi_id: DANDI dataset ID (e.g., "000055" for AJILE12)
            subjects: List of subject IDs to include
            tasks: List of task names to include
            runs: List of run numbers to include
            sessions: List of session IDs to include
            transform: Optional transform to apply to data
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.patch_size = patch_size
        self.stride = stride
        self.max_channels = max_channels
        self.ecog_amplitude_scale = ecog_amplitude_scale
        self.preload = preload
        self.use_mni_coords = use_mni_coords
        self.require_mni = require_mni
        self.data_format = data_format
        self.dandi_id = dandi_id
        self.transform = transform
        
        # Filter parameters
        self.subjects = subjects
        self.tasks = tasks
        self.runs = runs
        self.sessions = sessions
        
        # Derived parameters
        self.seq_length = int(seq_duration * sample_rate)
        self.num_patches = (self.seq_length - patch_size) // stride + 1
        
        # File list and metadata
        self.file_list = []
        self.file_metadata = []  # List of dicts with metadata
        self.channel_info = []  # List of channel metadata
        
        # Scan for data files
        self._scan_data_dir()
        
        if len(self.file_list) == 0:
            raise ValueError(f"No ECoG files found in {data_dir}")
        
        print(f"[ECoGDataset] Found {len(self.file_list)} ECoG files")
        
        # Preload if requested
        if self.preload:
            self.data_cache = []
            self._preload_data()
    
    def _scan_data_dir(self):
        """Scan data directory for ECoG files."""
        if self.dandi_id and DANDI_AVAILABLE:
            # DANDI dataset
            self._scan_dandi()
        else:
            # Local files
            self._scan_local_files()
    
    def _scan_dandi(self):
        """Scan DANDI archive for ECoG datasets."""
        try:
            from dandi.dandiapi import DandiAPIClient
            
            client = DandiAPIClient()
            dandiset = client.get_dandiset(self.dandi_id)
            
            # Get all NWB files
            for asset in dandiset.get_assets():
                if asset.path.endswith('.nwb'):
                    # Check if it matches filter criteria
                    metadata = asset.get_metadata()
                    
                    # Extract subject, task, run, session from metadata
                    subject = metadata.get('subject', {}).get('subject_id', 'unknown')
                    task = metadata.get('session', {}).get('task', 'unknown')
                    run = metadata.get('session', {}).get('run', 'unknown')
                    session = metadata.get('session', {}).get('session_id', 'unknown')
                    
                    # Apply filters
                    if self.subjects and subject not in self.subjects:
                        continue
                    if self.tasks and task not in self.tasks:
                        continue
                    if self.runs and run not in self.runs:
                        continue
                    if self.sessions and session not in self.sessions:
                        continue
                    
                    # Download or get local path
                    local_path = self._get_dandi_asset_path(asset)
                    if local_path:
                        self.file_list.append(str(local_path))
                        self.file_metadata.append({
                            'subject': subject,
                            'task': task,
                            'run': run,
                            'session': session,
                            'source': f'dandi:{self.dandi_id}',
                            'path': str(local_path),
                        })
            
            print(f"[ECoGDataset] Found {len(self.file_list)} files in DANDI:{self.dandi_id}")
            
        except Exception as e:
            warnings.warn(f"Failed to scan DANDI dataset {self.dandi_id}: {e}")
    
    def _get_dandi_asset_path(self, asset) -> Optional[Path]:
        """Get local path for DANDI asset, downloading if necessary."""
        # Check if file already exists locally
        local_dir = self.data_dir / f"dandi_{self.dandi_id}"
        local_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = local_dir / Path(asset.path).name
        
        if not local_path.exists():
            try:
                print(f"Downloading {asset.path} from DANDI...")
                asset.download(local_path)
                print(f"Downloaded to {local_path}")
            except Exception as e:
                warnings.warn(f"Failed to download {asset.path}: {e}")
                return None
        
        return local_path
    
    def _scan_local_files(self):
        """Scan local directory for ECoG files."""
        supported_formats = {
            '.nwb': 'nwb',
            '.edf': 'edf',
            '.bdf': 'edf',
            '.fif': 'fif',
            '.set': 'eeglab',
            '.vhdr': 'brainvision',
            '.eeg': 'brainvision',
            '.vmrk': 'brainvision',
        }
        
        for ext, format_name in supported_formats.items():
            if self.data_format != "auto" and format_name != self.data_format:
                continue
            
            files = list(self.data_dir.glob(f"**/*{ext}"))
            for fpath in files:
                if not fpath.is_file():
                    continue
                
                # Try to extract metadata from filename (BIDS convention)
                metadata = self._extract_bids_metadata(fpath)
                
                # Apply filters
                subject = metadata.get('subject', 'unknown')
                task = metadata.get('task', 'unknown')
                run = metadata.get('run', 'unknown')
                session = metadata.get('session', 'unknown')
                
                if self.subjects and subject not in self.subjects:
                    continue
                if self.tasks and task not in self.tasks:
                    continue
                if self.runs and run not in self.runs:
                    continue
                if self.sessions and session not in self.sessions:
                    continue
                
                self.file_list.append(str(fpath))
                self.file_metadata.append({
                    'subject': subject,
                    'task': task,
                    'run': run,
                    'session': session,
                    'source': 'local',
                    'path': str(fpath),
                    'format': format_name,
                })
    
    def _extract_bids_metadata(self, filepath: Path) -> Dict[str, str]:
        """Extract BIDS metadata from filename."""
        metadata = {}
        filename = filepath.name
        
        # BIDS pattern: sub-<subject>_ses-<session>_task-<task>_run-<run>_<modality>.<ext>
        parts = filename.split('_')
        for part in parts:
            if part.startswith('sub-'):
                metadata['subject'] = part[4:]
            elif part.startswith('ses-'):
                metadata['session'] = part[4:]
            elif part.startswith('task-'):
                metadata['task'] = part[5:]
            elif part.startswith('run-'):
                metadata['run'] = part[4:]
        
        return metadata
    
    def _preload_data(self):
        """Preload data into memory."""
        print(f"[ECoGDataset] Preloading {len(self.file_list)} files...")
        self.data_cache = []
        
        for i, fpath in enumerate(self.file_list):
            try:
                data, metadata = self._load_single_file(fpath, idx=i)
                if data is not None:
                    self.data_cache.append((data, metadata))
            except Exception as e:
                print(f"[ECoGDataset] Warning: Failed to load {fpath}: {e}")
        
        print(f"[ECoGDataset] Preloaded {len(self.data_cache)} files")
    
    def _load_single_file(
        self,
        filepath: str,
        idx: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Load a single ECoG file.
        
        Returns:
            data: (C, T) ECoG data or None if loading failed
            metadata: Dictionary with channel info, MNI coords, etc.
        """
        if not MNE_AVAILABLE:
            # Simplified loading without MNE
            return self._load_simple(filepath, idx)
        
        filepath = Path(filepath)
        metadata = self.file_metadata[idx] if idx is not None else {}
        
        try:
            # Determine file format
            if filepath.suffix == '.nwb':
                raw, montage, channel_info = load_dandi_nwb(str(filepath))
            elif filepath.suffix in ['.edf', '.bdf']:
                raw, montage, channel_info = load_edf_ecog(str(filepath))
            elif filepath.suffix == '.fif':
                raw = read_raw_fif(str(filepath), preload=False)
                montage = raw.get_montage()
                channel_info = self._extract_channel_info(raw)
            elif filepath.suffix == '.set':
                raw = read_raw_eeglab(str(filepath), preload=False)
                montage = raw.get_montage()
                channel_info = self._extract_channel_info(raw)
            else:
                # Try generic MNE loading
                raw = mne.io.read_raw(str(filepath), preload=False)
                montage = raw.get_montage()
                channel_info = self._extract_channel_info(raw)
            
            # Preprocess
            data, channel_info = preprocess_ecog(
                raw=raw,
                channel_info=channel_info,
                target_sfreq=self.sample_rate,
                max_channels=self.max_channels,
            )
            
            # Extract MNI coordinates if available
            mni_coords = None
            if self.use_mni_coords and montage is not None:
                mni_coords = extract_mni_coords(montage, channel_info)
                if mni_coords is None and self.require_mni:
                    print(f"[ECoGDataset] Skipping {filepath.name}: No MNI coordinates")
                    return None, {}
            
            # Get channel types (ECoG grid vs sEEG depth)
            channel_types = []
            channel_names = []
            for ch_info in channel_info:
                ch_name = ch_info.get('ch_name', f'Ch{len(channel_names)}')
                channel_names.append(ch_name)
                
                # Map channel type
                ctype = map_channel_type(ch_info)
                channel_types.append(ctype)
            
            # Prepare metadata
            metadata.update({
                'channel_names': channel_names,
                'channel_types': channel_types,
                'mni_coords': mni_coords,
                'num_channels': len(channel_names),
                'original_sfreq': raw.info['sfreq'],
                'duration': raw.times[-1] if hasattr(raw, 'times') else 0,
            })
            
            return data, metadata
            
        except Exception as e:
            print(f"[ECoGDataset] Error loading {filepath}: {e}")
            return None, {}
    
    def _load_simple(self, filepath: str, idx: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Simplified loading without MNE (for testing)."""
        filepath = Path(filepath)
        metadata = self.file_metadata[idx] if idx is not None else {}
        
        try:
            if filepath.suffix == '.npy':
                data = np.load(filepath)
            elif filepath.suffix == '.npz':
                npz = np.load(filepath)
                data = npz['data'] if 'data' in npz else npz[npz.files[0]]
            else:
                return None, {}
            
            # Simulate channel info
            C, T = data.shape if data.ndim == 2 else (1, data.shape[0])
            C = min(C, self.max_channels)
            
            # Truncate if needed
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = data[:C, :]
            
            # Generate dummy channel info
            channel_names = [f'Ch{i}' for i in range(C)]
            channel_types = [1] * C  # Assume all ECoG
            
            metadata.update({
                'channel_names': channel_names,
                'channel_types': channel_types,
                'mni_coords': None,
                'num_channels': C,
                'original_sfreq': self.sample_rate,
                'duration': T / self.sample_rate,
            })
            
            return data, metadata
            
        except Exception as e:
            print(f"[ECoGDataset] Simple loading failed for {filepath}: {e}")
            return None, {}
    
    def _extract_channel_info(self, raw) -> List[Dict[str, Any]]:
        """Extract channel information from MNE Raw object."""
        channel_info = []
        for idx, ch in enumerate(raw.info['chs']):
            ch_info = {
                'ch_name': ch['ch_name'],
                'kind': ch['kind'],
                'coord_frame': ch.get('coord_frame'),
                'loc': ch['loc'].tolist() if ch['loc'] is not None else None,
                'unit': ch['unit'],
                'cal': ch['cal'],
            }
            
            # Try to get additional info from montage
            montage = raw.get_montage()
            if montage is not None and ch['ch_name'] in montage.ch_names:
                ch_idx = montage.ch_names.index(ch['ch_name'])
                ch_info['montage_pos'] = montage.dig[ch_idx]['r'].tolist()
                ch_info['montage_kind'] = montage.dig[ch_idx]['kind']
            
            channel_info.append(ch_info)
        
        return channel_info
    
    def __len__(self) -> int:
        return len(self.file_list) if not self.preload else len(self.data_cache)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with:
            - eeg: (C, T) ECoG data
            - channel_names: List of C channel names
            - channel_types: (C,) integer tensor: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
            - mni_coords: (C, 3) MNI coordinates or zeros if not available
            - subject_id: Integer subject ID
            - metadata: Additional metadata
        """
        # Load data
        if self.preload:
            data, metadata = self.data_cache[idx]
        else:
            data, metadata = self._load_single_file(self.file_list[idx], idx)
        
        if data is None:
            # Return empty sample (should be filtered by DataLoader)
            return self._get_empty_sample()
        
        # Extract or generate channel names
        channel_names = metadata.get('channel_names', [f'Ch{i}' for i in range(data.shape[0])])
        
        # Channel types
        channel_types_np = np.array(metadata.get('channel_types', [1] * data.shape[0]), dtype=np.int64)
        channel_types = torch.from_numpy(channel_types_np)
        
        # MNI coordinates
        mni_coords = metadata.get('mni_coords')
        if mni_coords is None:
            mni_coords = np.zeros((len(channel_names), 3), dtype=np.float32)
        mni_coords = torch.from_numpy(mni_coords.astype(np.float32))
        
        # Subject ID (hash subject string to integer)
        subject_str = metadata.get('subject', 'unknown')
        subject_id = hash(subject_str) % 10000  # Limit to 10000 subjects
        
        # Ensure data is 2D (C, T)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Truncate or pad to seq_length
        C, T = data.shape
        if T < self.seq_length:
            # Pad with zeros
            pad_len = self.seq_length - T
            data = np.pad(data, ((0, 0), (0, pad_len)), mode='constant')
        else:
            # Truncate
            data = data[:, :self.seq_length]
        
        # Convert to torch tensor
        eeg = torch.from_numpy(data.astype(np.float32))
        
        # Apply amplitude normalization based on channel type
        eeg = self._normalize_amplitude(eeg, channel_types)
        
        # Apply transform if specified
        if self.transform is not None:
            eeg = self.transform(eeg)
        
        # Create sample dictionary
        sample = {
            'eeg': eeg,
            'channel_names': channel_names,
            'channel_types': channel_types,
            'mni_coords': mni_coords,
            'subject_id': torch.tensor(subject_id, dtype=torch.long),
            'metadata': metadata,
        }
        
        return sample
    
    def _normalize_amplitude(
        self,
        eeg: torch.Tensor,
        channel_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize amplitude based on channel type.
        ECoG signals are ~20× larger than scalp EEG.
        """
        if channel_types is None:
            return eeg
        
        # Create amplitude scaling factors
        # scalp_EEG: 1.0, ecog_grid: 1/20, seeg_depth: 1/10, unknown: 1.0
        scale_factors = torch.ones_like(channel_types, dtype=eeg.dtype)
        
        # ECoG: scale down by ecog_amplitude_scale
        ecog_mask = (channel_types == 1)
        scale_factors[ecog_mask] = 1.0 / self.ecog_amplitude_scale
        
        # sEEG: scale down by half of ECoG
        seeg_mask = (channel_types == 2)
        scale_factors[seeg_mask] = 1.0 / (self.ecog_amplitude_scale / 2)
        
        # Apply scaling
        scale_factors = scale_factors.view(1, -1, 1)  # (1, C, 1)
        eeg = eeg * scale_factors
        
        # Then z-score normalize per channel
        mean = eeg.mean(dim=-1, keepdim=True)
        std = eeg.std(dim=-1, keepdim=True) + 1e-6
        eeg = (eeg - mean) / std
        
        return eeg
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return an empty sample (for error handling)."""
        return {
            'eeg': torch.zeros((1, self.seq_length), dtype=torch.float32),
            'channel_names': ['Ch0'],
            'channel_types': torch.tensor([3], dtype=torch.long),  # unknown
            'mni_coords': torch.zeros((1, 3), dtype=torch.float32),
            'subject_id': torch.tensor(0, dtype=torch.long),
            'metadata': {'error': 'failed_to_load'},
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.preload:
            warnings.warn("Statistics require preload=True")
            return {}
        
        stats = {
            'num_files': len(self.data_cache),
            'num_channels_list': [],
            'durations': [],
            'channel_type_counts': {0: 0, 1: 0, 2: 0, 3: 0},
            'subjects': set(),
        }
        
        for data, metadata in self.data_cache:
            if data is None:
                continue
            
            stats['num_channels_list'].append(data.shape[0])
            stats['durations'].append(metadata.get('duration', 0))
            
            # Channel types
            channel_types = metadata.get('channel_types', [])
            for ctype in channel_types:
                if ctype in stats['channel_type_counts']:
                    stats['channel_type_counts'][ctype] += 1
            
            # Subjects
            stats['subjects'].add(metadata.get('subject', 'unknown'))
        
        # Compute averages
        if stats['num_channels_list']:
            stats['avg_channels'] = np.mean(stats['num_channels_list'])
            stats['min_channels'] = np.min(stats['num_channels_list'])
            stats['max_channels'] = np.max(stats['num_channels_list'])
        
        if stats['durations']:
            stats['avg_duration'] = np.mean(stats['durations'])
            stats['total_duration_hours'] = np.sum(stats['durations']) / 3600
        
        stats['num_subjects'] = len(stats['subjects'])
        
        return stats


class PairedECoGfMRIDataset(Dataset):
    """
    Dataset for paired ECoG-fMRI data (e.g., DANDI:000623).
    
    Provides synchronized ECoG and fMRI samples for cross-modal training.
    """
    
    def __init__(
        self,
        ecog_dataset: ECoGDataset,
        fmri_dataset: Dataset,  # Should be compatible with NeuroSTORM/BrainLM
        sync_tolerance: float = 0.1,  # Seconds
        require_sync: bool = True,
    ):
        """
        Initialize paired dataset.
        
        Args:
            ecog_dataset: ECoG dataset
            fmri_dataset: fMRI dataset (should have timestamps)
            sync_tolerance: Maximum time difference for sync (seconds)
            require_sync: Whether to require synchronized samples
        """
        self.ecog_dataset = ecog_dataset
        self.fmri_dataset = fmri_dataset
        self.sync_tolerance = sync_tolerance
        self.require_sync = require_sync
        
        # Build sync mapping
        self.sync_pairs = self._build_sync_pairs()
        
        if len(self.sync_pairs) == 0 and require_sync:
            raise ValueError("No synchronized ECoG-fMRI pairs found")
        
        print(f"[PairedECoGfMRIDataset] Found {len(self.sync_pairs)} synchronized pairs")
    
    def _build_sync_pairs(self) -> List[Tuple[int, int]]:
        """Build mapping between ECoG and fMRI samples based on timestamps."""
        sync_pairs = []
        
        # This is a simplified implementation
        # In practice, you would need access to timestamps in both datasets
        # For DANDI:000623, the sync is provided in the NWB file
        
        # For now, pair by index (assuming datasets are aligned)
        min_len = min(len(self.ecog_dataset), len(self.fmri_dataset))
        for i in range(min_len):
            sync_pairs.append((i, i))
        
        return sync_pairs
    
    def __len__(self) -> int:
        return len(self.sync_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get synchronized ECoG-fMRI pair."""
        ecog_idx, fmri_idx = self.sync_pairs[idx]
        
        # Get ECoG sample
        ecog_sample = self.ecog_dataset[ecog_idx]
        
        # Get fMRI sample
        fmri_sample = self.fmri_dataset[fmri_idx]
        
        # Combine samples
        sample = {
            'ecog': ecog_sample['eeg'],
            'ecog_channel_names': ecog_sample['channel_names'],
            'ecog_channel_types': ecog_sample['channel_types'],
            'ecog_mni_coords': ecog_sample['mni_coords'],
            'fmri': fmri_sample.get('fmri'),
            'fmri_roi': fmri_sample.get('roi'),
            'fmri_voxel': fmri_sample.get('voxel'),
            'subject_id': ecog_sample['subject_id'],
            'sync_idx': torch.tensor(idx, dtype=torch.long),
        }
        
        return sample


def test_ecog_dataset():
    """Test function for ECoGDataset."""
    import tempfile
    import shutil
    
    # Create temporary directory with dummy data
    temp_dir = tempfile.mkdtemp()
    print(f"Test directory: {temp_dir}")
    
    try:
        # Create dummy ECoG data
        C, T = 64, 5120  # 64 channels, 10 seconds at 512 Hz
        dummy_data = np.random.randn(C, T).astype(np.float32)
        
        # Save as numpy file
        test_file = Path(temp_dir) / "test_ecog.npy"
        np.save(test_file, dummy_data)
        
        # Test dataset
        dataset = ECoGDataset(
            data_dir=temp_dir,
            sample_rate=512,
            seq_duration=5.0,
            patch_size=256,
            stride=128,
            max_channels=256,
            preload=False,
            use_mni_coords=False,
            require_mni=False,
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"EEG shape: {sample['eeg'].shape}")
        print(f"Channel types: {sample['channel_types']}")
        print(f"Subject ID: {sample['subject_id']}")
        
        # Test statistics (requires preload)
        dataset2 = ECoGDataset(
            data_dir=temp_dir,
            preload=True,
        )
        stats = dataset2.get_statistics()
        print(f"Statistics: {stats}")
        
        print("\nECoG dataset test passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_ecog_dataset()