"""
Smoke test for the preprocessing pipeline.

Tests all GPU ops and the full EEG/fMRI pipelines with synthetic data.
Run: python -m brain_moe_pinn.preprocessing.test_preprocessing
"""

import sys
import torch
import numpy as np

from brain_moe_pinn.utils.device_utils import get_device
import warnings

warnings.filterwarnings("ignore")


def test_filtering():
    from brain_moe_pinn.preprocessing.gpu_ops.filtering import (
        design_fir_bandpass, design_fir_notch, apply_fir_filter,
    )
    kernel = design_fir_bandpass(0.5, 50.0, 256.0, numtaps=101)
    assert kernel.shape == (101,), f"Bad kernel shape: {kernel.shape}"

    data = torch.randn(2, 19, 2560)
    filtered = apply_fir_filter(data, kernel)
    assert filtered.shape == data.shape, f"Shape mismatch: {filtered.shape} vs {data.shape}"

    notch_kernel = design_fir_notch(50.0, 256.0, numtaps=101)
    assert notch_kernel.shape == (101,)

    print("  [PASS] Filtering")


def test_resampling():
    from brain_moe_pinn.preprocessing.gpu_ops.resampling import resample_1d, resample_3d
    data = torch.randn(2, 19, 2560)
    resampled = resample_1d(data, orig_freq=256.0, new_freq=200.0, method="interpolate")
    assert resampled.shape[0] == 2 and resampled.shape[1] == 19
    assert abs(resampled.shape[2] - 2000) < 10

    vol = torch.randn(64, 64, 64)
    resampled_vol = resample_3d(vol, target_shape=(32, 32, 32), mode="trilinear")
    assert resampled_vol.shape == (32, 32, 32)

    print("  [PASS] Resampling")


def test_spectral():
    from brain_moe_pinn.preprocessing.gpu_ops.spectral import (
        compute_psd, compute_bandpower, compute_spectral_slope,
    )
    data = torch.randn(2, 19, 2560)
    freqs, psd = compute_psd(data, sfreq=256.0, n_fft=256)
    assert psd.shape == (2, 19, 129), f"PSD shape: {psd.shape}"

    bands = compute_bandpower(data, sfreq=256.0, n_fft=256)
    assert "alpha" in bands and "gamma" in bands
    assert bands["alpha"].shape == (2, 19)

    slope = compute_spectral_slope(data, sfreq=256.0, n_fft=512)
    assert slope.shape == (2, 19)

    print("  [PASS] Spectral")


def test_spatial():
    from brain_moe_pinn.preprocessing.gpu_ops.spatial import (
        gaussian_kernel_3d, smooth_3d, slice_timing_correction,
    )
    kernel = gaussian_kernel_3d(sigma=2.0)
    assert kernel.sum().item() - 1.0 < 0.01

    vol = torch.randn(32, 32, 32)
    smoothed = smooth_3d(vol, fwhm=4.0, voxel_size=2.0)
    assert smoothed.shape == vol.shape

    bold = torch.randn(32, 32, 16, 100)
    slice_times = torch.linspace(0, 1.5, 16)
    corrected = slice_timing_correction(bold, slice_times, tr=2.0)
    assert corrected.shape == bold.shape

    print("  [PASS] Spatial")


def test_quality():
    from brain_moe_pinn.preprocessing.gpu_ops.quality import (
        detect_bad_channels, detect_bad_spans, full_quality_assessment,
    )
    data = torch.randn(19, 2560)
    bad_mask, details = detect_bad_channels(data)
    assert bad_mask.shape == (19,)

    correlated_data = torch.randn(1, 2560).expand(19, -1) + torch.randn(19, 1) * 0.1
    bad_mask_corr, _ = detect_bad_channels(correlated_data)
    assert bad_mask_corr.sum() < 19

    span_mask, span_details = detect_bad_spans(data, sfreq=256.0)
    assert span_mask.shape == (2560,)

    report = full_quality_assessment(correlated_data.unsqueeze(0), sfreq=256.0)
    assert 0.0 <= report.quality_score <= 1.0

    noisy_data = correlated_data.clone()
    noisy_data[0, :100] = 1000.0
    report_noisy = full_quality_assessment(noisy_data.unsqueeze(0), sfreq=256.0)
    assert report_noisy.quality_score < report.quality_score

    print("  [PASS] Quality")


def test_ica():
    from brain_moe_pinn.preprocessing.gpu_ops.ica import GPUFastICA
    torch.manual_seed(42)
    C, T = 10, 1000
    sources = torch.randn(C, T)
    mixing = torch.randn(C, C)
    data = mixing @ sources

    ica = GPUFastICA(n_components=C, max_iter=100, tol=1e-3)
    ica.fit(data)
    recovered = ica.transform(data)
    assert recovered.shape == (C, T)

    cleaned = ica.inverse_transform(recovered, exclude=[0, 1])
    assert cleaned.shape == (C, T)

    print("  [PASS] ICA")


def test_asr():
    from brain_moe_pinn.preprocessing.gpu_ops.asr import GPUASR
    torch.manual_seed(42)
    C, T = 10, 2560
    clean = torch.randn(C, T) * 10.0
    noisy = clean.clone()
    noisy[:, 1000:1200] += torch.randn(C, 200) * 500.0

    asr = GPUASR(sfreq=256.0, cutoff=-3.5)
    asr.fit(clean[:, :768])
    cleaned = asr.transform(noisy)
    assert cleaned.shape == noisy.shape

    print("  [PASS] ASR")


def test_parcellation():
    from brain_moe_pinn.preprocessing.gpu_ops.parcellation import (
        GPUParcellation, confound_regression, temporal_filter,
    )
    atlas = torch.zeros(32, 32, 16, dtype=torch.long)
    atlas[:16, :16, :8] = 1
    atlas[16:, :16, :8] = 2
    atlas[:16, 16:, :8] = 3
    atlas[16:, 16:, :8] = 4

    parcel = GPUParcellation("custom")
    parcel.load_atlas(atlas)
    assert parcel.n_rois == 4

    bold = torch.randn(32, 32, 16, 100)
    roi_ts = parcel.extract(bold)
    assert roi_ts.shape == (4, 100)

    roi_filtered = temporal_filter(roi_ts, tr=2.0, low_freq=0.008, high_freq=0.1)
    assert roi_filtered.shape == roi_ts.shape

    print("  [PASS] Parcellation")


def test_eeg_pipeline():
    from brain_moe_pinn.preprocessing.eeg_pipeline import (
        EEGPreprocessingPipeline, EEGPreprocessingConfig,
    )
    config = EEGPreprocessingConfig(
        target_sfreq=256,
        bandpass=(0.5, 50.0),
        notch_freqs=[50],
        artifact_mode="mask",
        normalization="zscore",
        target_montage=["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                        "T3", "C3", "Cz", "C4", "T4",
                        "T5", "P3", "Pz", "P4", "T6", "O1", "O2"],
        device=get_device(),
    )
    pipeline = EEGPreprocessingPipeline(config)

    data = torch.randn(19, 2560)
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "T3", "C3", "Cz", "C4", "T4",
                "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

    output = pipeline(data, sfreq=256.0, ch_names=ch_names)
    assert output.data.shape[0] == 19, f"Channels: {output.data.shape[0]}"
    assert output.channel_mask.shape[0] == 19
    assert output.span_mask.shape[0] > 0
    assert 0.0 <= output.quality_score <= 1.0
    assert output.sfreq == 256.0

    print("  [PASS] EEG Pipeline")


def test_eeg_pipeline_clinical():
    from brain_moe_pinn.preprocessing.eeg_pipeline import (
        EEGPreprocessingPipeline, EEGPreprocessingConfig,
    )
    config = EEGPreprocessingConfig(
        target_sfreq=256,
        bandpass=(0.5, 50.0),
        artifact_mode="repair",
        ica_components=5,
        normalization="zscore",
        target_montage=["Ch0", "Ch1", "Ch2", "Ch3", "Ch4"],
        device=get_device(),
    )
    pipeline = EEGPreprocessingPipeline(config)

    data = torch.randn(5, 2560)
    ch_names = ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4"]
    output = pipeline(data, sfreq=256.0, ch_names=ch_names)
    assert output.data.shape[0] == 5

    print("  [PASS] EEG Pipeline (clinical mode)")


def test_fmri_pipeline():
    from brain_moe_pinn.preprocessing.fmri_pipeline import (
        fMRIPreprocessingPipeline, fMRIPreprocessingConfig,
    )
    config = fMRIPreprocessingConfig(
        atlas="schaefer_400",
        tr=2.0,
        n_dummy_scans=2,
        slice_timing_correction=False,
        smoothing_fwhm=0.0,
        normalization="zscore",
        device=get_device(),
    )
    pipeline = fMRIPreprocessingPipeline(config)

    bold = torch.randn(16, 16, 8, 50)
    atlas = torch.zeros(16, 16, 8, dtype=torch.long)
    atlas[:8, :8, :4] = 1
    atlas[8:, :8, :4] = 2
    atlas[:8, 8:, :4] = 3
    atlas[8:, 8:, :4] = 4
    brain_mask = atlas > 0

    output = pipeline(bold, atlas_volume=atlas, brain_mask=brain_mask)
    assert output.roi_ts.shape[0] == 4
    assert output.roi_ts.shape[1] == 50 - 2
    assert output.tr == 2.0

    print("  [PASS] fMRI Pipeline")


def test_paired_pipeline():
    from brain_moe_pinn.preprocessing.paired_pipeline import PairedPreprocessor
    from brain_moe_pinn.preprocessing.eeg_pipeline import EEGPreprocessingConfig
    from brain_moe_pinn.preprocessing.fmri_pipeline import fMRIPreprocessingConfig

    eeg_config = EEGPreprocessingConfig(
        target_sfreq=256, bandpass=(0.5, 50.0), artifact_mode="mask",
        normalization="zscore", target_montage=["Ch0", "Ch1", "Ch2"],
        device=get_device(),
    )
    fmri_config = fMRIPreprocessingConfig(
        atlas="schaefer_400", tr=2.0, slice_timing_correction=False,
        smoothing_fwhm=0.0, normalization="zscore", device=get_device(),
    )

    paired = PairedPreprocessor(eeg_config, fmri_config)
    eeg = torch.randn(3, 2560)
    fmri = torch.randn(16, 16, 8, 50)
    atlas = torch.zeros(16, 16, 8, dtype=torch.long)
    atlas[:8, :8, :4] = 1
    atlas[8:, 8:, :4] = 2

    output = paired(eeg, fmri, fmri_tr=2.0, eeg_sfreq=256.0)
    assert output.eeg is not None
    assert output.fmri is not None
    assert output.overlapping_duration > 0

    print("  [PASS] Paired Pipeline")


def test_io_modules():
    from brain_moe_pinn.preprocessing.io.eeg_io import eeg_raw_to_tensor, EEGRawData
    from brain_moe_pinn.preprocessing.io.fmri_io import fmri_raw_to_tensor, fMRIRawData
    import numpy as np

    eeg_raw = EEGRawData(
        data=np.random.randn(19, 2560).astype(np.float32),
        sfreq=256.0,
        ch_names=[f"Ch{i}" for i in range(19)],
        ch_types=["eeg"] * 19,
        montage_3d=None,
        info={},
        file_path="test",
    )
    data, meta = eeg_raw_to_tensor(eeg_raw)
    assert data.shape == (19, 2560)
    assert meta["sfreq"] == 256.0

    fmri_raw = fMRIRawData(
        bold=np.random.randn(16, 16, 8, 50).astype(np.float32),
        affine=np.eye(4),
        header=None,
        tr=2.0,
        shape=(16, 16, 8, 50),
        slice_times=None,
        info={},
        file_path="test",
    )
    bold, meta = fmri_raw_to_tensor(fmri_raw)
    assert bold.shape == (16, 16, 8, 50)
    assert meta["tr"] == 2.0

    print("  [PASS] I/O modules")


def run_all_tests():
    print("Preprocessing Pipeline Smoke Tests")
    print("=" * 50)

    tests = [
        ("Filtering", test_filtering),
        ("Resampling", test_resampling),
        ("Spectral", test_spectral),
        ("Spatial", test_spatial),
        ("Quality", test_quality),
        ("ICA", test_ica),
        ("ASR", test_asr),
        ("Parcellation", test_parcellation),
        ("I/O modules", test_io_modules),
        ("EEG Pipeline (foundation)", test_eeg_pipeline),
        ("EEG Pipeline (clinical)", test_eeg_pipeline_clinical),
        ("fMRI Pipeline", test_fmri_pipeline),
        ("Paired Pipeline", test_paired_pipeline),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
