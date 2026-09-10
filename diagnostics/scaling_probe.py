"""Scaling-honesty probes (torch-native): predictability ceiling + 3-scale ablation.

Before committing to large-scale training, measure (a) the intrinsic
predictability ceiling of the signal (information horizon), and (b) whether
model quality saturates with scale on YOUR evaluation axes.  Both are cheap
and run on existing data; they decide where scale helps (representation)
vs where structure/objectives are the bottleneck (dynamics).

All heavy computation is torch-native and device-agnostic (pass CPU or GPU
tensors); numpy is used only for tiny offline reductions (single scalars).
"""

import numpy as np
import torch


def _as_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def information_horizon(x, fs=1.0, thresh=0.5, max_lag=None, device=None):
    """First lag (in seconds) where mean autocorrelation drops below thresh.

    x: (T,) or (T, C) signal (tensor or array).  A hard ceiling for pointwise
    prediction: beyond this horizon no model family can track the signal.
    """
    z = _as_tensor(x)
    if z.ndim == 1:
        z = z[:, None]
    z = z.to(device)
    z = (z - z.mean(0)) / (z.std(0) + 1e-9)
    T, C = z.shape
    max_lag = max_lag or min(T - 2, 5000)
    n_c = min(C, 200)
    ac = torch.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        a = z[:-lag, :n_c]
        b = z[lag:, :n_c]
        ac[lag] = (a * b).mean()
    below = torch.where(ac[1:].abs() < thresh)[0]
    if below.numel() == 0:
        return float(max_lag) / fs
    return (int(below[0]) + 1) / fs


def spectral_floor(x, fs=1.0, p_low=0.2, p_high=0.8, device=None):
    """Fraction of variance in the (p_low, p_high) frequency band.

    Signals dominated by DC/low frequency are ill-suited to frequency-domain
    methods and statistically simpler to track; report this before choosing
    the model family.
    """
    x = _as_tensor(x)
    if x.ndim == 1:
        x = x[:, None]
    x = x.to(device) - x.mean(0)
    n = x.shape[0]
    X = torch.fft.rfft(x, dim=0)
    ps = X.abs() ** 2
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs, device=x.device)
    tot = ps.sum(0) + 1e-12
    band = ps[(freqs >= p_low) & (freqs <= p_high)].sum(0) / tot
    return float(band.mean())


def variance_maintenance(real, generated, tail_frac=0.2, device=None):
    """std(generated)/std(real) and tail/first std ratio of the rollout."""
    r = _as_tensor(real).to(device)
    g = _as_tensor(generated).to(device)
    ratio = float((g.std(0).mean() / (r.std(0).mean() + 1e-9)).item())
    k = max(1, int(tail_frac * g.shape[0]))
    tail = float((g[-k:].std(0).mean() / (g[:k].std(0).mean() + 1e-9)).item())
    return {"std_ratio": ratio, "tail_decay": tail}


def run_scale_ablation(model_factory, scale_factors, evaluate, seed=0):
    """Generic 3-scale ablation: does quality saturate before the largest scale?

    model_factory(scale) -> fitted generative model (any family)
    evaluate(model)      -> dict of metrics (higher = better per axis)
    Returns per-scale metric dicts + a saturation flag per metric
    (gain from the last doubling < 10% => saturated).
    """
    results = {}
    for sc in scale_factors:
        model = model_factory(sc)
        results[sc] = evaluate(model)
    sat = {}
    scales = list(scale_factors)
    for metric in results[scales[0]]:
        vals = [results[s][metric] for s in scales]
        if len(vals) >= 2:
            gain = abs(vals[-1] - vals[-2]) / (abs(vals[-2]) + 1e-9)
            sat[metric] = bool(gain < 0.10)
    return {"per_scale": results, "saturated": sat,
            "note": "numpy is used here only for scalar bookkeeping; "
                    "model/evaluate loops are user-supplied (use torch)."}


if __name__ == "__main__":
    # self-test on synthetic signals (torch)
    torch.manual_seed(0)
    t = torch.arange(4000, dtype=torch.float32)
    slow = (torch.sin(2 * np.pi * 0.01 * t) + 0.2 * torch.randn(4000))[:, None]
    white = torch.randn(4000, 1)
    print("slow info horizon:", information_horizon(slow, fs=1.0))
    print("white info horizon:", information_horizon(white, fs=1.0))
    print("slow spectral floor:", round(spectral_floor(slow), 3))
