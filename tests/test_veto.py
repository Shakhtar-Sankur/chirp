"""Offline tests of the glitch vetoes: chi-square on signals and glitches, reweighted SNR, gating."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chirp import search, taylorf2, veto  # noqa: E402

FS, T, F_LOW, P = 1024.0, 32.0, 20.0, 16
N = int(FS * T)
f = np.fft.rfftfreq(N, 1 / FS)
DF = 1 / T
INV = np.where(f >= F_LOW, FS / 2.0, 0.0)  # white noise of unit variance: S(f) = 2 / fs
H = taylorf2.waveform(f, 10.0, 10.0)
SIGMA = np.sqrt(4 * DF * np.sum(np.abs(H) ** 2 * INV))


def signal(snr, at):
    return np.fft.irfft(H * snr / SIGMA * np.exp(-2j * np.pi * f * at), n=N) * FS


def sine_gaussian(at, f0=150.0, q=8.0, amp=1.0):
    t = np.arange(N) / FS - at
    tau = q / (2 * np.pi * f0)
    return amp * np.exp(-(t / tau) ** 2) * np.sin(2 * np.pi * f0 * t)


def rho_chisq(x):
    return veto.snr_chisq(np.fft.rfft(x) / FS, H, INV, DF, P)


def test_bins_split_sigma_equally():
    e = veto.bin_edges(H, INV, P)
    w = np.abs(H) ** 2 * INV
    parts = np.array([w[e[k]:e[k + 1]].sum() for k in range(P)]) / w.sum()
    assert np.allclose(parts, 1 / P, atol=0.01)


def test_rho_matches_matched_filter():
    x = np.random.default_rng(1).standard_normal(N) + signal(10.0, 20.0)
    rho, _ = rho_chisq(x)
    ref, _ = search.snr_series(np.fft.rfft(x) / FS, H[None, :], INV, DF)
    assert np.allclose(rho, ref[0], rtol=1e-6, atol=1e-6)


def test_chisq_follows_its_distribution_for_a_signal():
    # at a matching signal's peak chi^2 is pure noise: mean 2p - 2 over noise realisations
    rng = np.random.default_rng(2)
    vals = []
    for _ in range(40):
        rho, chisq = rho_chisq(rng.standard_normal(N) + signal(20.0, 20.0))
        vals.append(chisq[int(np.argmax(rho))])
    assert abs(np.mean(vals) / (2 * P - 2) - 1) < 0.15


def test_glitch_is_downweighted_and_signal_is_not():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(N) + signal(12.0, 12.0) + sine_gaussian(24.0, amp=40.0)
    rho, chisq = rho_chisq(x)
    hat = veto.newsnr(rho, chisq, P)
    sig = slice(int(11.9 * FS), int(12.1 * FS))
    gl = slice(int(23.5 * FS), int(24.5 * FS))
    assert rho[gl].max() > rho[sig].max()          # before the veto the glitch is louder
    assert hat[sig].max() > 0.9 * rho[sig].max()   # the signal keeps its SNR
    assert hat[gl].max() < 0.5 * hat[sig].max()    # the glitch no longer competes


def test_gating_removes_a_glitch_and_spares_a_signal():
    rng = np.random.default_rng(4)
    big = 5000.0 * np.exp(-((np.arange(N) / FS - 24.0) / 0.003) ** 2)
    x = rng.standard_normal(N) + signal(12.0, 12.0) + big
    pf, psd = search.welch_psd(x, FS)
    gates = veto.find_gates(x, FS, pf, psd, F_LOW)
    assert len(gates) == 1 and abs(gates[0] - 24.0) < 0.01
    y = veto.apply_gates(x, FS, gates)
    assert np.all(y[int(23.76 * FS):int(24.24 * FS)] == 0)
    assert np.array_equal(y[: int(23.4 * FS)], x[: int(23.4 * FS)])
    rho, _ = rho_chisq(y)
    assert rho.max() == pytest.approx(rho[int(11.9 * FS):int(12.1 * FS)].max())


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(5).standard_normal(N) + signal(10.0, 20.0)
    d = np.fft.rfft(x) / FS
    rho, chisq = rho_chisq(x)
    rt, ct = veto.snr_chisq(torch.tensor(d, dtype=torch.complex128), torch.tensor(H, dtype=torch.complex128),
                            torch.tensor(INV), DF, P, xp=torch)
    assert np.allclose(rt.numpy(), rho, rtol=1e-6, atol=1e-6)
    assert np.allclose(ct.numpy(), chisq, rtol=1e-5, atol=1e-4)


def test_gpu_clusters_and_rescore_on_cpu():
    torch = pytest.importorskip("torch")
    from chirp import gpu

    rng = np.random.default_rng(6)
    xH = rng.standard_normal(N) + signal(10.0, 20.0)
    xL = rng.standard_normal(N) + signal(8.0, 20.005) + sine_gaussian(9.0, amp=40.0)
    dH, dL = (torch.tensor(np.fft.rfft(x) / FS, dtype=torch.complex64) for x in (xH, xL))
    inv = torch.tensor(INV, dtype=torch.float32)
    hs = torch.tensor(np.stack([H, taylorf2.waveform(f, 14.0, 14.0)]), dtype=torch.complex64)
    first, length, count = int(4.5 * FS), int(FS), 24
    cv, ct, ci, nv, ni = gpu.batch_network_snr_clusters(dH, dL, inv, inv, hs, DF, FS, 0.012, first, length, count,
                                                        (int(19 * FS), int(21 * FS)))
    # numpy reference for the loudest cluster value
    k = int(round(0.012 * FS))
    from scipy.ndimage import maximum_filter1d
    rH, _ = search.snr_series(np.fft.rfft(xH) / FS, hs.numpy().astype(complex), INV, DF)
    rL, _ = search.snr_series(np.fft.rfft(xL) / FS, hs.numpy().astype(complex), INV, DF)
    net = np.sqrt(rH**2 + maximum_filter1d(rL, size=2 * k + 1, axis=-1, mode="constant") ** 2)
    ref = net[:, first:first + count * length].reshape(2, count, length).max(axis=(0, 2))
    assert np.allclose(cv.numpy(), ref, rtol=1e-4)
    j = 15  # the cluster holding 20 s
    assert abs(ci[j].item() / FS - 20.0) < 0.02 and ct[j].item() == 0
    rho, rhat = gpu.rescore_network(dH, dL, inv, inv, hs[0], DF, FS, 0.012, P)
    sig, gl = slice(int(19.5 * FS), int(20.5 * FS)), slice(int(8.5 * FS), int(9.5 * FS))
    assert rhat[sig].max() > 0.9 * rho[sig].max()
    assert rhat[gl].max() < 0.6 * rhat[sig].max()
