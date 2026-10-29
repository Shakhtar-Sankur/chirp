"""Telling signals from glitches: gating and the chi-square discriminator.

Gating. Instrumental glitches are short and orders of magnitude louder than any astrophysical
signal in whitened data. As in the PyCBC searches, samples of the whitened strain above
`threshold` standard deviations mark a glitch, and the raw strain is zeroed for `width` seconds
either side, with Hann tapers of `taper` seconds, before the noise spectrum is estimated.

Chi-square (Allen, Phys. Rev. D 71, 062001, 2005). The template's band is split into p bins that
each carry 1/p of its sigma^2. For a signal that matches the template, each bin's complex SNR
zeta_k(t) is close to zeta(t) / p; a glitch puts its power in a few bins. With
zeta(t) = z(t) / sigma (so rho = |zeta|),

    chi^2(t) = p sum_k |zeta_k(t) - zeta(t)/p|^2 = p sum_k |zeta_k(t)|^2 - |zeta(t)|^2,

which in Gaussian noise follows a chi-square distribution with 2p - 2 degrees of freedom.

Reweighted SNR (Babak et al., Phys. Rev. D 87, 024033, 2013), the ranking statistic of the
PyCBC searches: with the reduced chi-square chi_r^2 = chi^2 / (2p - 2),

    rho_hat = rho                                   if chi_r^2 <= 1
    rho_hat = rho / [(1 + chi_r^6) / 2]^(1/6)       otherwise.

`xp` is the array module: numpy, or torch for tensors on a GPU.
"""

import numpy as np

from . import search


def bin_edges(h, inv_psd, p):
    """Frequency indices [e_0 = 0, ..., e_p = F] splitting the template's sigma^2 into p equal parts."""
    w = np.abs(h) ** 2 * inv_psd
    c = np.cumsum(w)
    e = np.searchsorted(c, c[-1] * np.arange(1, p) / p)
    return np.concatenate([[0], e, [w.size]])


def snr_chisq(dtilde, h, inv_psd, df, p=16, xp=np):
    """rho(t) and chi^2(t) for one template h (F,). Returns two (N,) arrays, N = 2 (F - 1)."""
    if xp is np:
        edges = bin_edges(h, inv_psd, p)
        F = h.size
        bins = np.zeros((p, F), dtype=complex)
        for k in range(p):
            bins[k, edges[k]:edges[k + 1]] = h[edges[k]:edges[k + 1]]
    else:
        w = (h.real**2 + h.imag**2) * inv_psd
        c = xp.cumsum(w.double(), dim=0)
        cut = xp.searchsorted(c, c[-1] * xp.arange(1, p, device=h.device, dtype=c.dtype) / p)
        k_of_f = xp.bucketize(xp.arange(h.shape[0], device=h.device), cut, right=True)
        bins = xp.zeros((p, h.shape[0]), dtype=h.dtype, device=h.device)
        bins[k_of_f, xp.arange(h.shape[0], device=h.device)] = h
    # every bin is filtered with the full template's sigma, so zeta_k sums to zeta
    sigma = xp.sqrt(4.0 * df * xp.sum((h.real**2 + h.imag**2) * inv_psd))
    F = h.shape[-1]
    N = 2 * (F - 1)
    if xp is np:
        full = np.zeros((p, N), dtype=complex)
        full[:, :F] = 4.0 * df * dtilde * bins.conj() * inv_psd
        zk = np.fft.ifft(full, axis=-1) * N / sigma
    else:
        full = xp.zeros((p, N), dtype=h.dtype, device=h.device)
        full[:, :F] = 4.0 * df * dtilde * bins.conj() * inv_psd
        zk = xp.fft.ifft(full, dim=-1) * N / sigma
    z = zk.sum(0)
    rho2 = z.real**2 + z.imag**2
    chisq = p * (zk.real**2 + zk.imag**2).sum(0) - rho2
    return xp.sqrt(rho2), chisq


def newsnr(rho, chisq, p=16, xp=np):
    """Reweighted SNR: rho down-weighted where the reduced chi-square exceeds 1."""
    r = chisq / (2 * p - 2)
    w = ((1.0 + r**3) / 2.0) ** (1.0 / 6.0)
    return xp.where(r > 1.0, rho / w, rho)


def find_gates(x, fs, psd_f, psd, f_low, threshold=50.0, width=0.25):
    """Glitch times (s from the start of x): peaks of the whitened strain above `threshold` sigma.

    Peaks closer than 2 * width are merged into the loudest one.
    """
    n = x.size
    f = np.fft.rfftfreq(n, 1.0 / fs)
    inv = search.inverse_psd(psd_f, psd, f, fs, f_low)
    xw = np.fft.irfft(np.fft.rfft(x * search.tukey(n, 0.02)) * np.sqrt(inv), n=n)
    edge = int(4 * fs)  # the taper and the whitening filter corrupt the ends
    core = xw[edge:n - edge]
    sd = 1.4826 * np.median(np.abs(core - np.median(core)))
    loud = np.flatnonzero(np.abs(core) > threshold * sd) + edge
    gates = []
    for i in loud[np.argsort(-np.abs(xw[loud]))]:
        if all(abs(i - g) > 2 * width * fs for g in gates):
            gates.append(int(i))
    return sorted(g / fs for g in gates)


def apply_gates(x, fs, gates, width=0.25, taper=0.25):
    """x zeroed within `width` s of each gate time, with Hann tapers of `taper` s outside that."""
    y = x.copy()
    m = int(taper * fs)
    ramp = 0.5 * (1 + np.cos(np.pi * np.arange(1, m + 1) / m))  # 1 -> 0 going inwards
    for g in gates:
        a, b = int((g - width) * fs), int((g + width) * fs)
        y[max(a, 0):max(min(b, y.size), 0)] = 0.0
        for j in range(m):
            if 0 <= a - 1 - j < y.size:
                y[a - 1 - j] *= ramp[m - 1 - j]
            if 0 <= b + j < y.size:
                y[b + j] *= ramp[m - 1 - j]
    return y
