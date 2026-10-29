"""The search on a GPU: one template batch at a time, every event's data held whitened in memory.

For each event and template the two detectors' SNR series are combined into a coincident network
SNR: rho_net(t) = sqrt(rho_H(t)^2 + max_{|d| <= window} rho_L(t + d)^2), the window covering the
10 ms light travel time between the LIGO sites plus timing slack. Kept per event: the best value
near the catalog time (targeted) and over the whole valid stretch of data (blind).
"""

import numpy as np
import torch
import torch.nn.functional as F


def to_device(dtilde, inv_psd, band, device):
    """Data spectrum and inverse PSD as float32-safe tensors.

    1/S(f) is ~1e46 for LIGO strain, past float32's range. Scaling 1/S by c and d by 1/sqrt(c)
    leaves every SNR unchanged (rho goes as c / sqrt(c) * 1/sqrt(c) = 1), so c brings 1/S to ~1.
    """
    c = 1.0 / np.median(inv_psd[band])
    d = torch.tensor(dtilde / np.sqrt(c), dtype=torch.complex64, device=device)
    return d, torch.tensor(inv_psd * c, dtype=torch.float32, device=device)


def templates_to_device(hs, device):
    """Each template scaled to unit peak amplitude (SNR does not depend on a template's scale)."""
    hs = np.asarray(hs)
    peak = np.abs(hs).max(axis=1, keepdims=True)
    return torch.tensor(hs / np.where(peak > 0, peak, 1.0), dtype=torch.complex64, device=device)


def batch_network_snr(dH, dL, invH, invL, hs, df, fs, window_s, valid, near):
    """dH, dL: (F,) complex64 data spectra; invH, invL: (F,) float32 inverse PSDs;
    hs: (B, F) complex64 templates. valid, near: (start, stop) sample ranges.

    Returns per template: (blind max, blind index, near max, near index, rhoH max near, rhoL max near).
    """
    Fn = dH.shape[-1]
    N = 2 * (Fn - 1)

    def rho(d, inv):
        w = hs.conj() * inv
        sigma = torch.sqrt(4.0 * df * torch.sum((hs.real**2 + hs.imag**2) * inv, dim=-1))
        full = torch.zeros((hs.shape[0], N), dtype=hs.dtype, device=hs.device)
        full[:, :Fn] = 4.0 * df * d * w
        return torch.abs(torch.fft.ifft(full, dim=-1)) * N / sigma[:, None]

    rH, rL = rho(dH, invH), rho(dL, invL)
    k = int(round(window_s * fs))
    rLmax = F.max_pool1d(rL[:, None, :], kernel_size=2 * k + 1, stride=1, padding=k)[:, 0, :]
    net = torch.sqrt(rH**2 + rLmax**2)
    out = []
    for a, b in (valid, near):
        v, i = net[:, a:b].max(dim=-1)
        out += [v, i + a]
    out += [rH[:, near[0]:near[1]].max(dim=-1).values, rL[:, near[0]:near[1]].max(dim=-1).values]
    return out


def batch_network_snr_clusters(dH, dL, invH, invL, hs, df, fs, window_s, first, length, count, near):
    """As batch_network_snr, but the blind search keeps the loudest template in each of `count`
    clusters of `length` samples starting at sample `first`, for the chi-square pass to rescore.

    Returns (cluster max (count,), its template in the batch (count,), its sample (count,),
    near max (B,), near index (B,)).
    """
    Fn = dH.shape[-1]
    N = 2 * (Fn - 1)

    def rho(d, inv):
        w = hs.conj() * inv
        sigma = torch.sqrt(4.0 * df * torch.sum((hs.real**2 + hs.imag**2) * inv, dim=-1))
        full = torch.zeros((hs.shape[0], N), dtype=hs.dtype, device=hs.device)
        full[:, :Fn] = 4.0 * df * d * w
        return torch.abs(torch.fft.ifft(full, dim=-1)) * N / sigma[:, None]

    rH, rL = rho(dH, invH), rho(dL, invL)
    k = int(round(window_s * fs))
    rLmax = F.max_pool1d(rL[:, None, :], kernel_size=2 * k + 1, stride=1, padding=k)[:, 0, :]
    net = torch.sqrt(rH**2 + rLmax**2)
    cl = net[:, first:first + count * length].reshape(net.shape[0], count, length)
    v, i = cl.max(dim=-1)                       # (B, count)
    cv, ct = v.max(dim=0)                       # (count,)
    ci = i[ct, torch.arange(count, device=net.device)] + first + torch.arange(count, device=net.device) * length
    nv, ni = net[:, near[0]:near[1]].max(dim=-1)
    return cv, ct, ci, nv, ni + near[0]


def rescore_network(dH, dL, invH, invL, h, df, fs, window_s, p=16):
    """Network SNR and network reweighted SNR series (N,) for one template h (F,) complex64.

    Each detector's SNR is reweighted by its own chi-square before the two are combined.
    """
    from . import veto

    rH, cH = veto.snr_chisq(dH, h, invH, df, p, xp=torch)
    rL, cL = veto.snr_chisq(dL, h, invL, df, p, xp=torch)
    k = int(round(window_s * fs))

    def net(a, b):
        bm = F.max_pool1d(b[None, None, :], kernel_size=2 * k + 1, stride=1, padding=k)[0, 0]
        return torch.sqrt(a**2 + bm**2)

    return net(rH, rL), net(veto.newsnr(rH, cH, p, xp=torch), veto.newsnr(rL, cL, p, xp=torch))
