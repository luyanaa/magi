"""
Brain MoE-PINN Data Preprocessing Pipeline.

GPU-accelerated preprocessing for EEG and fMRI signals:
- EEG: 6-stage pipeline (I/O → conditioning → quality → artifact → normalize → montage)
- fMRI: 4-stage pipeline (I/O → spatial → temporal → parcellation)
- Paired: Cross-modal alignment for simultaneous EEG-fMRI

Two modes:
- "foundation" (default): minimal preprocessing, mask bad data, match field standard
- "clinical": full artifact removal (ICA, ASR, bad channel interpolation)

All GPU operations use torch-native primitives (F.conv1d, torchaudio, torch.stft, etc.)
"""

from .eeg_pipeline import (
    EEGPreprocessingPipeline,
    EEGPreprocessingConfig,
    EEGPreprocessedOutput,
    STANDARD_10_20,
    CHANNEL_3D_POSITIONS,
)

from .fmri_pipeline import (
    fMRIPreprocessingPipeline,
    fMRIPreprocessingConfig,
    fMRIPreprocessedOutput,
)

from .paired_pipeline import (
    PairedPreprocessor,
    PairedOutput,
)
