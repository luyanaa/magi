"""
GPU-accelerated FastICA for EEG artifact removal.

Implements FastICA using torch.linalg primitives for GPU execution.
Designed for EOG/ECG/EMG artifact removal in clinical mode (Mode B).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class GPUFastICA:
    """GPU-accelerated FastICA using torch.linalg primitives.

    Usage:
        ica = GPUFastICA(n_components=20)
        ica.fit(data)  # data: (C, T) on GPU
        sources = ica.transform(data)
        cleaned = ica.inverse_transform(sources, exclude=[0, 3])  # remove components 0 and 3
    """

    def __init__(
        self,
        n_components: int = 20,
        max_iter: int = 200,
        tol: float = 1e-4,
        fun: str = "logcosh",
        alpha: float = 1.0,
        whiten: bool = True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.alpha = alpha
        self.whiten = whiten

        self.mixing_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.mean_: Optional[torch.Tensor] = None
        self.whitening_: Optional[torch.Tensor] = None
        self.unmixing_: Optional[torch.Tensor] = None

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        """Nonquadratic contrast function."""
        if self.fun == "logcosh":
            return torch.tanh(self.alpha * x)
        elif self.fun == "exp":
            return x * torch.exp(-x ** 2 / 2.0)
        elif self.fun == "cube":
            return x ** 3
        else:
            return torch.tanh(self.alpha * x)

    def _g_prime(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of contrast function."""
        if self.fun == "logcosh":
            return self.alpha * (1 - torch.tanh(self.alpha * x) ** 2)
        elif self.fun == "exp":
            return (1 - x ** 2) * torch.exp(-x ** 2 / 2.0)
        elif self.fun == "cube":
            return 3 * x ** 2
        else:
            return self.alpha * (1 - torch.tanh(self.alpha * x) ** 2)

    def fit(self, X: torch.Tensor) -> "GPUFastICA":
        """Fit FastICA model.

        Args:
            X: (C, T) signal tensor on GPU (C channels, T time samples)

        Returns:
            self
        """
        C, T = X.shape
        n_comp = min(self.n_components, C)

        self.mean_ = X.mean(dim=1, keepdim=True)
        X_centered = X - self.mean_

        if self.whiten:
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            K = U[:, :n_comp] / (S[:n_comp].unsqueeze(0) + 1e-10)
            K = K.T  # (n_comp, C)
            X_white = K @ X_centered  # (n_comp, T)
            self.whitening_ = K
        else:
            X_white = X_centered[:n_comp]
            self.whitening_ = torch.eye(n_comp, C, device=X.device, dtype=X.dtype)

        W = torch.zeros(n_comp, n_comp, device=X.device, dtype=X.dtype)

        for i in range(n_comp):
            w = torch.randn(n_comp, device=X.device, dtype=X.dtype)
            w = w / (w.norm() + 1e-10)

            for iteration in range(self.max_iter):
                g_wx = self._g(w @ X_white)
                g_prime_wx = self._g_prime(w @ X_white)

                w_new = (X_white @ g_wx) / T - g_prime_wx.mean() * w

                if i > 0:
                    proj = w_new @ W[:i].T
                    w_new = w_new - proj @ W[:i]

                w_new = w_new / (w_new.norm() + 1e-10)

                convergence = min(
                    (w_new - w).abs().max().item(),
                    (w_new + w).abs().max().item(),
                )
                w = w_new

                if convergence < self.tol:
                    break

            W[i] = w

        self.components_ = W  # (n_comp, n_comp) in whitened space
        self.unmixing_ = W @ self.whitening_  # (n_comp, C) full unmixing
        self.mixing_ = torch.linalg.pinv(self.unmixing_)  # (C, n_comp)

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Extract independent components.

        Args:
            X: (C, T) signal tensor

        Returns:
            S: (n_components, T) independent components
        """
        X_centered = X - self.mean_
        return self.unmixing_ @ X_centered

    def inverse_transform(
        self,
        S: torch.Tensor,
        exclude: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Reconstruct signal from sources, optionally excluding components.

        Args:
            S: (n_components, T) source signals
            exclude: list of component indices to zero out (artifact components)

        Returns:
            X_recon: (C, T) reconstructed signal
        """
        S_filtered = S.clone()
        if exclude is not None:
            for idx in exclude:
                if 0 <= idx < S.shape[0]:
                    S_filtered[idx] = 0.0

        return self.mixing_ @ S_filtered + self.mean_

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def apply(
        self,
        X: torch.Tensor,
        exclude: List[int],
    ) -> torch.Tensor:
        """Convenience: fit, remove specified components, return cleaned signal."""
        S = self.fit_transform(X)
        return self.inverse_transform(S, exclude=exclude)


def classify_ica_components(
    sources: torch.Tensor,
    eeg_data: torch.Tensor,
    sfreq: float = 256.0,
    eog_threshold: float = 0.8,
    ecg_threshold: float = 0.6,
) -> Dict[str, List[int]]:
    """Heuristic classification of ICA components into artifact types.

    This is a simplified classifier. For production use, consider ICLabel.

    Args:
        sources: (n_comp, T) ICA component time courses
        eeg_data: (C, T) original EEG data
        sfreq: sampling rate
        eog_threshold: correlation threshold for EOG classification
        ecg_threshold: correlation threshold for ECG classification

    Returns:
        dict with 'eog', 'ecg', 'brain' component indices
    """
    n_comp = sources.shape[0]

    frontal_channels = None
    if eeg_data.shape[0] >= 3:
        frontal_channels = eeg_data[:3]  # rough approximation

    eog_components = []
    ecg_components = []
    brain_components = []

    for i in range(n_comp):
        comp = sources[i]

        if frontal_channels is not None:
            corr = torch.corrcoef(torch.stack([comp, frontal_channels.mean(dim=0)]))[0, 1]
            if abs(corr.item()) > eog_threshold:
                eog_components.append(i)
                continue

        spectrum = torch.fft.rfft(comp)
        power = (spectrum.abs() ** 2)
        freqs = torch.fft.rfftfreq(comp.shape[0], d=1.0 / sfreq)

        low_freq_power = power[freqs < 5.0].sum()
        total_power = power.sum() + 1e-12
        low_freq_ratio = low_freq_power / total_power

        if low_freq_ratio > 0.7:
            eog_components.append(i)
        else:
            brain_components.append(i)

    return {
        "eog": eog_components,
        "ecg": ecg_components,
        "brain": brain_components,
    }
