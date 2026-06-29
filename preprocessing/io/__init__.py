"""
I/O modules for loading raw neuroimaging data.

All I/O is CPU-based. Data is transferred to GPU after loading.
"""

from .eeg_io import (
    load_edf,
    load_nwb,
    load_bids,
    load_numpy,
    eeg_raw_to_tensor,
    EEGRawData,
)

from .fmri_io import (
    load_nifti,
    load_bids_fmri,
    load_fmriprep_derivatives,
    fmri_raw_to_tensor,
    fMRIRawData,
)

from .nwb_io import (
    load_dandi_nwb,
    nwb_to_tensor,
    NWBData,
)
