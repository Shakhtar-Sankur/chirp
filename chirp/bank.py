"""A non-spinning template bank for binary black holes, placed by measured match.

Templates are laid along chirp mass for a few mass ratios. Each step down in chirp mass is the
largest whose template still matches the previous one at `min_match` (overlap maximised over
time and phase, weighted by a real detector noise spectrum), found by bisection. This is a
simple one-dimensional placement, not an optimal lattice: it errs dense.
"""

import numpy as np

QS = (1.0, 1.5, 2.5, 4.0, 6.0, 9.0)


def masses(mc, q):
    """Component masses (m1 >= m2) from chirp mass and mass ratio q = m1 / m2 >= 1."""
    eta = q / (1 + q) ** 2
    m = mc / eta**0.6
    return m * q / (1 + q), m / (1 + q)


def imr_template(m1, m2, f, f_low):
    """IMRPhenomXAS (non-spinning, (2,2) mode) on the one-sided grid f, from LALSimulation."""
    import lal
    import lalsimulation as ls

    df = f[1] - f[0]
    hp, _ = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0, 1e6 * lal.PC_SI,
                                           0, 0, 0, 0, 0, df, f_low, f[-1], f_low, lal.CreateDict(), ls.IMRPhenomXAS)
    h = np.zeros(f.size, complex)
    n = min(hp.data.length, f.size)
    h[:n] = hp.data.data[:n]
    return h


def match(h1, h2, inv_psd):
    """Overlap of two templates maximised over relative time and phase."""
    a = h1 * h2.conj() * inv_psd
    z = np.abs(np.fft.ifft(np.concatenate([a, np.zeros(a.size - 2)])))
    n1 = np.sqrt(np.sum(np.abs(h1) ** 2 * inv_psd))
    n2 = np.sqrt(np.sum(np.abs(h2) ** 2 * inv_psd))
    return z.max() * (2 * a.size - 2) / (n1 * n2)


def build(f, inv_psd, f_low, mc_min, mc_max, min_match=0.97, qs=QS, template=imr_template):
    """List of (m1, m2, mc, q), chirp mass descending, for each mass ratio."""
    out = []
    for q in qs:
        mc = mc_max
        prev = template(*masses(mc, q), f, f_low)
        out.append((*masses(mc, q), mc, q))
        while mc > mc_min:
            lo, hi = 0.80, 0.9999  # ratio of the next chirp mass to this one
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if match(prev, template(*masses(mc * mid, q), f, f_low), inv_psd) >= min_match:
                    hi = mid
                else:
                    lo = mid
            mc *= hi
            prev = template(*masses(mc, q), f, f_low)
            out.append((*masses(mc, q), mc, q))
    return out
