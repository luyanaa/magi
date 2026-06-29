"""
Paired EEG-fMRI preprocessing pipeline with temporal alignment.

Handles simultaneous recordings (CineBrain, DANDI:000623) and
provides temporal alignment via sync markers.
"""

import torch
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass

from .eeg_pipeline import EEGPreprocessingPipeline, EEGPreprocessedOutput, EEGPreprocessingConfig
from .fmri_pipeline import fMRIPreprocessingPipeline, fMRIPreprocessedOutput, fMRIPreprocessingConfig


@dataclass
class PairedOutput:
    eeg: EEGPreprocessedOutput
    fmri: fMRIPreprocessedOutput
    alignment_quality: float
    time_offset_eeg_to_fmri: float
    overlapping_duration: float
    metadata: Dict


class PairedPreprocessor:
    """Joint preprocessing of paired EEG-fMRI recordings with temporal alignment.

    Usage:
        eeg_config = EEGPreprocessingConfig()
        fmri_config = fMRIPreprocessingConfig()
        paired = PairedPreprocessor(eeg_config, fmri_config)
        output = paired(eeg_raw, fmri_raw, sync_markers=...)
    """

    def __init__(
        self,
        eeg_config: Optional[EEGPreprocessingConfig] = None,
        fmri_config: Optional[fMRIPreprocessingConfig] = None,
    ):
        self.eeg_pipeline = EEGPreprocessingPipeline(eeg_config)
        self.fmri_pipeline = fMRIPreprocessingPipeline(fmri_config)

    def __call__(
        self,
        eeg_raw,
        fmri_raw,
        sync_markers: Optional[Dict] = None,
        eeg_sfreq: Optional[float] = None,
        fmri_tr: Optional[float] = None,
    ) -> PairedOutput:
        """Process paired EEG-fMRI recording.

        Args:
            eeg_raw: raw EEG data (EEGRawData, tensor, or numpy)
            fmri_raw: raw fMRI data (fMRIRawData, tensor, or path)
            sync_markers: dict with sync timing info:
                - 'eeg_triggers': list of EEG sample indices where scanner pulses detected
                - 'fmri_triggers': list of fMRI volume indices (usually 0,1,2,...)
                - 'eeg_trigger_times': list of seconds
                - 'fmri_trigger_times': list of seconds
            eeg_sfreq: EEG sampling rate (if not in raw data)
            fmri_tr: fMRI TR (if not in raw data)

        Returns:
            PairedOutput with aligned EEG and fMRI outputs
        """
        eeg_out = self.eeg_pipeline(eeg_raw, sfreq=eeg_sfreq)
        fmri_out = self.fmri_pipeline(fmri_raw)

        if sync_markers is not None:
            time_offset, overlap_duration, quality = self._compute_alignment(sync_markers, eeg_out, fmri_out)
        else:
            time_offset = 0.0
            overlap_duration = min(
                eeg_out.data.shape[-1] / eeg_out.sfreq,
                fmri_out.roi_ts.shape[-1] * fmri_out.tr,
            )
            quality = 0.5

        # --- Temporal cropping to overlapping window ---
        eeg_out, fmri_out = self._crop_to_overlap(
            eeg_out, fmri_out, time_offset, overlap_duration
        )

        return PairedOutput(
            eeg=eeg_out,
            fmri=fmri_out,
            alignment_quality=quality,
            time_offset_eeg_to_fmri=time_offset,
            overlapping_duration=overlap_duration,
            metadata={
                "sync_markers": sync_markers is not None,
                "eeg_sfreq": eeg_out.sfreq,
                "fmri_tr": fmri_out.tr,
            },
        )

    def _crop_to_overlap(
        self,
        eeg_out: EEGPreprocessedOutput,
        fmri_out: fMRIPreprocessedOutput,
        time_offset: float,
        overlap_duration: float,
    ) -> Tuple[EEGPreprocessedOutput, fMRIPreprocessedOutput]:
        """Crop EEG and fMRI to their overlapping temporal window.

        time_offset > 0 means fMRI starts after EEG.
        """
        if overlap_duration <= 0:
            return eeg_out, fmri_out

        sfreq = eeg_out.sfreq
        tr = fmri_out.tr

        # EEG crop indices
        eeg_start_sec = max(0.0, time_offset)
        eeg_end_sec = eeg_start_sec + overlap_duration
        eeg_start_idx = int(eeg_start_sec * sfreq)
        eeg_end_idx = min(int(eeg_end_sec * sfreq), eeg_out.data.shape[-1])

        # fMRI crop indices
        fmri_start_sec = max(0.0, -time_offset)
        fmri_end_sec = fmri_start_sec + overlap_duration
        fmri_start_idx = int(fmri_start_sec / tr)
        fmri_end_idx = min(int(fmri_end_sec / tr), fmri_out.roi_ts.shape[-1])

        if eeg_end_idx > eeg_start_idx:
            eeg_out.data = eeg_out.data[:, eeg_start_idx:eeg_end_idx]
            if eeg_out.span_mask is not None and eeg_out.span_mask.numel() > 0:
                eeg_out.span_mask = eeg_out.span_mask[eeg_start_idx:eeg_end_idx]

        if fmri_end_idx > fmri_start_idx:
            fmri_out.roi_ts = fmri_out.roi_ts[:, fmri_start_idx:fmri_end_idx]

        return eeg_out, fmri_out

    def _compute_alignment(
        self,
        sync_markers: Dict,
        eeg_out: EEGPreprocessedOutput,
        fmri_out: fMRIPreprocessedOutput,
    ) -> Tuple[float, float, float]:
        """Compute temporal alignment between EEG and fMRI.

        Returns:
            time_offset: seconds from EEG time 0 to fMRI time 0
            overlap_duration: seconds of overlapping recording
            quality: alignment quality (0-1)
        """
        eeg_triggers = sync_markers.get("eeg_trigger_times", sync_markers.get("eeg_triggers", []))
        fmri_triggers = sync_markers.get("fmri_trigger_times", sync_markers.get("fmri_triggers", []))

        if not eeg_triggers or not fmri_triggers:
            return 0.0, 0.0, 0.0

        if len(eeg_triggers) > 1 and len(fmri_triggers) > 1:
            eeg_interval = eeg_triggers[1] - eeg_triggers[0]
            fmri_interval = fmri_triggers[1] - fmri_triggers[0]

            interval_ratio = min(eeg_interval, fmri_interval) / (max(eeg_interval, fmri_interval) + 1e-10)
            quality = interval_ratio
        else:
            quality = 0.5

        time_offset = 0.0
        if eeg_triggers and fmri_triggers:
            time_offset = fmri_triggers[0] - eeg_triggers[0]

        eeg_duration = eeg_out.data.shape[-1] / eeg_out.sfreq
        fmri_duration = fmri_out.roi_ts.shape[-1] * fmri_out.tr

        eeg_end = eeg_duration
        fmri_start = time_offset
        fmri_end = time_offset + fmri_duration

        overlap_start = max(0.0, fmri_start)
        overlap_end = min(eeg_end, fmri_end)
        overlap_duration = max(0.0, overlap_end - overlap_start)

        if eeg_duration > 0 and fmri_duration > 0:
            quality *= min(1.0, overlap_duration / min(eeg_duration, fmri_duration))

        return time_offset, overlap_duration, quality
