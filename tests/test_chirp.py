"""Offline tests: TaylorF2 against LALSimulation, the matched filter on a synthetic signal, the bank."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chirp import bank, search, taylorf2  # noqa: E402


def test_taylorf2_matches_lalsimulation():
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    df, f_low = 1 / 64, 20.0
    for m1, m2 in [(1.4, 1.4), (10.0, 1.4), (20.0, 15.0)]:
        hp, _ = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0, 1e8 * lal.PC_SI,
                                               0, 0, 0, 0, 0, df, f_low, 2048.0, f_low, lal.CreateDict(), ls.TaylorF2)
        f = np.arange(hp.data.length) * df
        mine = taylorf2.waveform(f, m1, m2)
        band = (f >= f_low) & (np.abs(mine) > 0)
        dphi = np.unwrap(np.angle(hp.data.data[band]) - np.angle(mine[band]))
        A = np.vstack([np.ones(band.sum()), f[band]]).T
        res = dphi - A @ np.linalg.lstsq(A, dphi, rcond=None)[0]  # remove t_c and phi_c
        assert np.max(np.abs(res)) < 1e-8
        ratio = np.abs(hp.data.data[band]) / np.abs(mine[band])
        assert ratio.std() / ratio.mean() < 1e-12


def test_taylorf2_newtonian_limit():
    # at leading order Psi = 3/(128 eta v^5); the 1PN term is a small correction at low v
    f = np.array([10.0])
    m1 = m2 = 1.4
    eta = 0.25
    v = np.cbrt(np.pi * (m1 + m2) * taylorf2.MSUN_S * f)
    lead = 3 / (128 * eta * v**5)
    assert abs(taylorf2.phase(f, m1, m2)[0] / lead - 1) < 0.05


def test_matched_filter_recovers_injected_snr():
    rng = np.random.default_rng(0)
    fs, T, f_low = 1024.0, 32.0, 20.0
    n = int(fs * T)
    f = np.fft.rfftfreq(n, 1 / fs)
    psd = np.full(f.size, 2.0 / fs)  # white noise of unit variance
    noise = rng.standard_normal(n)
    h = taylorf2.waveform(f, 10.0, 10.0)
    inv = np.where(f >= f_low, 1.0 / psd, 0.0)
    df = 1 / T
    sigma = np.sqrt(4 * df * np.sum(np.abs(h) ** 2 * inv))
    target = 12.0
    shift = 20.0  # put coalescence 20 s into the segment
    sig = np.fft.irfft(h * target / sigma * np.exp(-2j * np.pi * f * shift), n=n) * fs
    dtilde = np.fft.rfft(noise + sig) / fs
    rho, _ = search.snr_series(dtilde, h[None, :], inv, df)
    k = int(np.argmax(rho[0]))
    assert abs(k / fs - shift) < 2 / fs
    assert abs(rho[0, k] - target) < 1.5  # noise moves the peak by about 1


def test_bank_masses_round_trip():
    for mc, q in [(10.0, 1.0), (30.0, 4.0)]:
        m1, m2 = bank.masses(mc, q)
        assert m1 >= m2
        assert abs(m1 / m2 - q) < 1e-12
        assert abs(taylorf2.chirp_mass(m1, m2) - mc) < 1e-12
