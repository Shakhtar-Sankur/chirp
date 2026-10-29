"""TaylorF2: the stationary-phase frequency-domain waveform of a compact binary's inspiral.

General relativity, solved order by order in v = (pi M f)^(1/3): the post-Newtonian energy and
flux give the phase of each frequency component in closed form, to 3.5PN (v^7) beyond the
leading quadrupole term. Point particles, non-spinning, (2,2) mode only, Newtonian amplitude.

    h(f) = A f^(-7/6) exp(-i Psi(f)),   0 < f <= f_ISCO = 1 / (6^(3/2) pi M)

    Psi(f) = 2 pi f t_c - phi_c - pi/4 + 3 / (128 eta v^5) sum_k (a_k + b_k ln v) v^k

Coefficients a_k, b_k: Blanchet, Living Rev. Relativ. 17, 2 (2014), Eq. (317), in the form
used by LALSimulation's TaylorF2 (Buonanno, Iyer, Ochsner, Pan & Sathyaprakash 2009, Table I),
which this module is checked against in tests/test_taylorf2.py.
"""

import numpy as np

MSUN_S = 4.925490947641267e-06  # G Msun / c^3 in seconds (IAU 2015 nominal solar mass parameter)
EULER_GAMMA = 0.5772156649015329


def phasing_coefficients(eta):
    """(a, b): a[k] and b[k] multiply v^k and v^k ln v inside the sum, k = 0..7."""
    pi = np.pi
    a = np.zeros(8)
    b = np.zeros(8)
    a[0] = 1.0
    a[2] = 3715.0 / 756.0 + 55.0 / 9.0 * eta
    a[3] = -16.0 * pi
    a[4] = 15293365.0 / 508032.0 + 27145.0 / 504.0 * eta + 3085.0 / 72.0 * eta**2
    a[5] = pi * (38645.0 / 756.0 - 65.0 / 9.0 * eta)
    b[5] = 3.0 * a[5]
    a[6] = (11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * pi**2 - 6848.0 / 21.0 * EULER_GAMMA
            + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * pi**2)
            + 76055.0 / 1728.0 * eta**2 - 127825.0 / 1296.0 * eta**3
            - 6848.0 / 21.0 * np.log(4.0))
    b[6] = -6848.0 / 21.0
    a[7] = pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta**2)
    return a, b


def f_isco(m1, m2):
    """Gravitational-wave frequency at the Schwarzschild innermost stable circular orbit (Hz)."""
    return 1.0 / (6.0**1.5 * np.pi * (m1 + m2) * MSUN_S)


def phase(f, m1, m2):
    """Psi(f) without the t_c, phi_c and -pi/4 terms (radians), masses in solar masses."""
    M = (m1 + m2) * MSUN_S
    eta = m1 * m2 / (m1 + m2) ** 2
    a, b = phasing_coefficients(eta)
    v = np.cbrt(np.pi * M * f)
    lv = np.log(v)
    s = np.zeros_like(v)
    for k in range(7, -1, -1):  # Horner in v, log terms carried alongside
        s = s * v + (a[k] + b[k] * lv)
    return 3.0 / (128.0 * eta * v**5) * s


def waveform(f, m1, m2):
    """Unit-amplitude-scale template h(f) = f^(-7/6) exp(-i Psi(f)) up to f_ISCO, zero above.

    The overall amplitude (distance, inclination, chirp mass) cancels in a normalised matched
    filter, so only the shape is returned. Frequencies f <= 0 give zero.
    """
    f = np.asarray(f, dtype=np.float64)
    h = np.zeros(f.shape, dtype=np.complex128)
    ok = (f > 0) & (f <= f_isco(m1, m2))
    psi = phase(f[ok], m1, m2) - np.pi / 4
    h[ok] = f[ok] ** (-7.0 / 6.0) * np.exp(-1j * psi)
    return h


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
