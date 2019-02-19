""" Analytic approximations for Schwarzschild QNMs.

The approximations implemented in this module can be used as initial
guesses when numerically searching for QNM frequencies.

"""

from __future__ import division, print_function, absolute_import

import numpy as np

def dolan_ottewill_expansion(s, l, n):
    """ High l asymptotic expansion of Schwarzschild QNM frequency.

    The result of [1]_ is an expansion in inverse powers of L =
    (l+1/2). Their paper stated this series out to L^{-4}, which is
    how many terms are implemented here. The coefficients in this
    series are themselves positive powers of N = (n+1/2). This means
    the expansion breaks down for large N.

    Parameters
    ----------
    s: int
      Spin weight of the field of interest.

    l: int
      Multipole number of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    n: int
      Overtone number of interest.

    Returns
    -------
    complex
      Analytic approximation of QNM of interest.

    References
    ----------
    .. [1] SR Dolan, AC Ottewill, "On an expansion method for black
       hole quasinormal modes and Regge poles," CQG 26 225003 (2009),
       https://arxiv.org/abs/0908.0329 .
    """

    L    = l + 0.5
    N    = n + 0.5
    beta = 1. - s*s

    DO_coeffs = { -1: 1.,
                   0: -1.j * N,
                  +1: beta/3. - 5.*N*N/36. - 115./432.,
                  +2: -1.j * N * (beta/9. + 235.*N*N/3888. - 1415./15552.),
                  +3: (-beta*beta/27. + (204.*N*N + 211.)/3888.*beta
                       +(854160.*N**4 - 1664760.*N*N - 776939.)/40310784.),
                  +4: -1.j * N *(beta*beta/27. + (1100.*N*N-2719.)/46656.*beta
                                 + (11273136.*N**4 - 52753800.*N*N + 66480535.)/2902376448.) }

    omega = 0.
    for k, c in DO_coeffs.items():
        omega = omega + c * np.power(L, -k)

    omega = omega / np.sqrt(27)

    return omega

def large_overtone_expansion(s, l, n):
    r""" The eikonal approximation for QNMs, valid for l >> n >> 1 .

    This is just the first two terms of the series in
    :meth:`dolan_ottewill_expansion`.
    The earliest work I know deriving this result is [1]_ but there
    may be others. In the eikonal approximation, valid when
    :math:`l \gg n \gg 1`, the QNM frequency is

    .. math:: \sqrt{27} M \omega \approx (l+\frac{1}{2}) - i (n+\frac{1}{2}) .

    Parameters
    ----------
    s: int
      Spin weight of the field of interest.

    l: int
      Multipole number of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    n: int
      Overtone number of interest.

    Returns
    -------
    complex
      Analytic approximation of QNM of interest.

    References
    ----------
    .. [1] V Ferrari, B Mashhoon, "New approach to the quasinormal
       modes of a black hole," Phys. Rev. D 30, 295 (1984)

    """

    k = np.log(3.)/(8. * np.pi)
    kappa = 0.25 # Surface gravity

    return k - 1.j * kappa * (n + 0.5)

def Schw_QNM_estimate(s, l, n):
    """ Give either :meth:`large_overtone_expansion` or :meth:`dolan_ottewill_expansion`.

    The Dolan-Ottewill expansion includes terms with higher powers of
    the overtone number n, so it breaks down faster at high n.

    Parameters
    ----------
    s: int
      Spin weight of the field of interest.

    l: int
      Multipole number of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    n: int
      Overtone number of interest.

    Returns
    -------
    complex
      Analytic approximation of QNM of interest.

    """

    if (( n > 3 ) and (n >= 2*l)):
        return large_overtone_expansion(s, l, n)
    else:
        return dolan_ottewill_expansion(s, l, n)
