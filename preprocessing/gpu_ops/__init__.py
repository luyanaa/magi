"""
GPU operations for brain signal preprocessing.

All operations are torch-native and support GPU execution.
"""

from .filtering import (
    design_fir_bandpass,
    design_fir_lowpass,
    design_fir_highpass,
    design_fir_notch,
    design_notch_kernel_set,
    apply_fir_filter,
    apply_iir_filter,
    apply_iir_filtfilt,
)

from .resampling import (
    resample_1d,
    resample_3d,
    resample_fmri_4d,
)

from .spectral import (
    compute_stft,
    compute_psd,
    compute_bandpower,
    compute_spectral_slope,
    compute_cross_frequency_coupling,
    BAND_DEFINITIONS,
)

from .spatial import (
    gaussian_kernel_3d,
    smooth_3d,
    smooth_4d_fmri,
    apply_affine_transform,
    slice_timing_correction,
)

from .quality import (
    detect_bad_channels,
    detect_bad_spans,
    compute_quality_score,
    full_quality_assessment,
    QualityReport,
)

from .ica import (
    GPUFastICA,
    classify_ica_components,
)

from .asr import GPUASR

from .parcellation import (
    GPUParcellation,
    confound_regression,
    temporal_filter,
    ATLAS_REGISTRY,
)

from .registration import GPURegistration
