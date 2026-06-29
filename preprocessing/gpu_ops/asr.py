"""
GPU-accelerated Artifact Subspace Reconstruction (ASR) for EEG.

Port of asrpy to torch. ASR identifies high-variance signal subspaces
and reconstructs them from clean reference data.

Reference: Mullen et al. (2015). Real-time neuroimaging and cognitive monitoring
using wearable dry EEG. IEEE TBME.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class GPUASR:
    """GPU-accelerated Artifact Subspace Reconstruction.

    Usage:
        asr = GPUASR(sfreq=256, cutoff=-3.5)
        asr.fit(clean_data)  # (C, T) clean reference segment
        cleaned = asr.transform(raw_data)  # (C, T) full recording
    """

    def __init__(
        self,
        sfreq: float = 256.0,
        cutoff: float = -3.5,
        block_size: int = 10,
        win_len: float = 0.5,
        win_overlap: float = 0.66,
        rank_delta: float = 0.02,
    ):
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.block_size = block_size
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.rank_delta = rank_delta

        self._clean_cov = None
        self._clean_mean = None
        self._n_channels = None

    def fit(self, X: torch.Tensor) -> "GPUASR":
        """Fit ASR model on clean reference data.

        Args:
            X: (C, T) clean reference EEG segment (at least 30s recommended)

        Returns:
            self
        """
        C, T = X.shape
        self._n_channels = C
        self._clean_mean = X.mean(dim=1, keepdim=True)
        X_centered = X - self._clean_mean

        self._clean_cov = self._estimate_covariance(X_centered)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply ASR to remove artifacts.

        Args:
            X: (C, T) raw EEG data

        Returns:
            X_clean: (C, T) artifact-corrected EEG data
        """
        if self._clean_cov is None:
            raise RuntimeError("ASR model not fitted. Call fit() first.")

        C, T = X.shape
        X_centered = X - self._clean_mean

        win_samples = int(self.win_len * self.sfreq)
        step = max(int(win_samples * (1 - self.win_overlap)), 1)

        X_clean = X_centered.clone()
        n_reconstructed = 0

        for start in range(0, T - win_samples + 1, step):
            end = start + win_samples
            window = X_centered[:, start:end]

            window_cov = self._estimate_covariance(window)

            is_artifact = self._detect_artifact_subspace(
                window_cov, self._clean_cov
            )

            if is_artifact:
                reconstructed = self._reconstruct_subspace(
                    window, self._clean_cov
                )
                X_clean[:, start:end] = reconstructed
                n_reconstructed += 1

        X_clean = X_clean + self._clean_mean

        return X_clean

    def fit_transform(self, X: torch.Tensor, clean_start: int = 0, clean_end: Optional[int] = None) -> torch.Tensor:
        """Fit on a clean segment and transform the full recording."""
        if clean_end is None:
            clean_end = min(int(30 * self.sfreq), X.shape[1])
        clean_data = X[:, clean_start:clean_end]
        self.fit(clean_data)
        return self.transform(X)

    def _estimate_covariance(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate regularized covariance matrix."""
        C, T = X.shape
        cov = (X @ X.T) / (T - 1)

        diag = cov.diag()
        reg = self.rank_delta * diag.mean()
        cov = cov + reg * torch.eye(C, device=X.device, dtype=X.dtype)

        return cov

    def _detect_artifact_subspace(
        self,
        window_cov: torch.Tensor,
        clean_cov: torch.Tensor,
    ) -> bool:
        """Detect if window contains artifact subspace.

        Compares eigenvalue distribution of window covariance
        against clean reference using the cutoff threshold.
        """
        eigvals_window = torch.linalg.eigvalsh(window_cov)
        eigvals_clean = torch.linalg.eigvalsh(clean_cov)

        log_window = torch.log(eigvals_window[eigvals_window > 0] + 1e-10)
        log_clean = torch.log(eigvals_clean[eigvals_clean > 0] + 1e-10)

        if len(log_window) == 0 or len(log_clean) == 0:
            return False

        window_mean = log_window.mean()
        clean_mean = log_clean.mean()
        clean_std = log_clean.std() + 1e-10

        z_score = (window_mean - clean_mean) / clean_std

        return z_score > abs(self.cutoff)

    def _reconstruct_subspace(
        self,
        X_window: torch.Tensor,
        clean_cov: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct artifact subspace using clean reference covariance.

        Projects data onto the clean covariance eigenspace, removing
        high-variance artifact directions.
        """
        C, T = X_window.shape

        eigvals, eigvecs = torch.linalg.eigh(clean_cov)

        eigvals_sorted, sort_idx = eigvals.sort(descending=True)
        eigvecs_sorted = eigvecs[:, sort_idx]

        total_var = eigvals_sorted.sum()
        cumvar = eigvals_sorted.cumsum(0) / (total_var + 1e-12)

        n_keep = max(1, (cumvar < (1 - self.rank_delta)).sum().item() + 1)
        n_keep = min(n_keep, C)

        V_keep = eigvecs_sorted[:, :n_keep]

        projected = V_keep.T @ X_window
        reconstructed = V_keep @ projected

        return reconstructed
