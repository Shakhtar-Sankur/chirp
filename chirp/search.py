"""Matched filtering: noise spectrum, whitening and the SNR time series.

For data d(t) with one-sided noise power spectral density S(f), the matched-filter output of a
template h is

    z(t) = 4 df sum_f d(f) h*(f) / S(f) exp(2 pi i f t),   sigma^2 = 4 df sum_f |h(f)|^2 / S(f)

and the signal-to-noise ratio rho(t) = |z(t)| / sigma maximises over the template's phase. In
stationary Gaussian noise rho^2 follows a chi-square distribution with 2 degrees of freedom.

`xp` is the array module: numpy on a CPU, or torch (with its fft) on a GPU.
"""

import numpy as np


def welch_psd(x, fs, seg_seconds=4.0):
    """One-sided PSD by the median of Hann-windowed periodograms, 50% overlap.

    The median resists glitches; dividing by ln 2 corrects its bias for exponential periodogram
    values (the large-number limit of the bias, as in common search pipelines).
    """
    n = int(seg_seconds * fs)
    step = n // 2
    w = np.hanning(n)
    norm = fs * np.sum(w**2)
    segs = np.lib.stride_tricks.sliding_window_view(x, n)[::step]
    p = np.abs(np.fft.rfft((segs - segs.mean(axis=1, keepdims=True)) * w, axis=1)) ** 2 / norm
    p[:, 1:-1] *= 2.0
    return np.fft.rfftfreq(n, 1.0 / fs), np.median(p, axis=0) / np.log(2.0)


def inverse_psd(psd_f, psd, f, fs, f_low, trunc_seconds=4.0):
    """1/S(f) on the grid f, zero below f_low, with inverse-spectrum truncation.

    Truncating sqrt(1/S) to trunc_seconds in the time domain stops narrow spectral lines from
    making the whitening filter (and so each template's response) arbitrarily long.
    """
    s = np.interp(f, psd_f, psd)
    inv = np.where((f >= f_low) & (s > 0), 1.0 / s, 0.0)
    n = 2 * (f.size - 1)
    q = np.fft.irfft(np.sqrt(inv), n=n)
    m = int(trunc_seconds * fs)
    taper = np.zeros(n)
    half = m // 2
    win = np.hanning(m)
    taper[:half] = win[half:]
    taper[-half:] = win[:half]
    inv_t = np.abs(np.fft.rfft(q * taper, n=n)) ** 2
    inv_t[f < f_low] = 0.0
    return inv_t


def tukey(n, alpha):
    w = np.ones(n)
    k = int(alpha * n / 2)
    if k > 0:
        r = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
        w[:k] = r
        w[-k:] = r[::-1]
    return w


def snr_series(dtilde, hs, inv_psd, df, xp=np):
    """rho(t) for each template.

    dtilde: (F,) complex data spectrum on the one-sided grid; hs: (B, F) complex templates;
    inv_psd: (F,) real. Returns (B, N) real SNR on the segment's time samples, N = 2 (F - 1),
    and sigma (B,).
    """
    w = hs.conj() * inv_psd
    sigma = xp.sqrt(4.0 * df * xp.sum((hs.real**2 + hs.imag**2) * inv_psd, axis=-1))
    F = dtilde.shape[-1]
    N = 2 * (F - 1)
    full = xp.zeros((hs.shape[0], N), dtype=hs.dtype) if xp is np else xp.zeros((hs.shape[0], N), dtype=hs.dtype, device=hs.device)
    full[:, :F] = 4.0 * df * dtilde * w
    # positive frequencies only: z(t) = sum_f full(f) exp(2 pi i f t) = N * ifft(full)
    z = xp.fft.ifft(full, axis=-1) * N if xp is np else xp.fft.ifft(full, dim=-1) * N
    return xp.abs(z) / sigma[:, None], sigma


def coincident_network_snr(rho_a, rho_b, fs, window_seconds):
    """Max over t of sqrt(rho_a(t)^2 + max_{|d| <= window} rho_b(t + d)^2), per template.

    rho_a, rho_b: (B, N) numpy arrays on the same time grid. Returns (snr, index_in_a).
    """
    w = int(round(window_seconds * fs))
    from scipy.ndimage import maximum_filter1d

    rb = maximum_filter1d(rho_b, size=2 * w + 1, axis=-1, mode="constant")
    net = np.sqrt(rho_a**2 + rb**2)
    i = np.argmax(net, axis=-1)
    return net[np.arange(net.shape[0]), i], i
