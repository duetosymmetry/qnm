""" Utilities for locating avoided crossings between QNM spin sequences.

Pairs of Kerr quasinormal modes can come very close to each other as the
spin varies. A well known example is the (s=-2, l=2, m=2, n=5) and n=6
pair near a ~ 0.9. The gap minima of neighboring overtone pairs and the
related exceptional points are an active topic, see for example
arXiv:2604.24584 and arXiv:2509.06411. Near a gap minimum, extracting the
two modes from ringdown data becomes badly conditioned, see for example
the numerical study in https://github.com/maiconburn/recoverability-criticality
(DOI 10.5281/zenodo.22156019), so knowing where the minimum is and how
small the gap gets is useful when planning or interpreting fits.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from .cached import KerrSeqCache

_cache = None


def _get_cache():
    global _cache
    if _cache is None:
        import qnm as _qnm
        _cache = _qnm.modes_cache
    return _cache


def mode_gap(a, mode_1, mode_2, interp_only=True):
    """Distance in the complex plane between two QNM frequencies at spin a.

    Parameters
    ----------
    a: float or array_like
      Dimensionless spin(s), 0 <= a < 1.

    mode_1: tuple
      Mode labels (s, l, m, n) of the first mode.

    mode_2: tuple
      Mode labels (s, l, m, n) of the second mode.

    interp_only: bool, optional [default: True]
      Evaluate frequencies from the precomputed spin-sequence interpolants
      instead of re-solving at each spin. Near an avoided crossing the
      root-polishing solver can fail to converge, which is the very
      phenomenon this module maps, so interpolation is the robust default.

    Returns
    -------
    float or ndarray
      |omega_1(a) - omega_2(a)| evaluated at each spin.
    """

    cache = _get_cache()
    seq_1 = cache(*mode_1)
    seq_2 = cache(*mode_2)

    def _gap(spin):
        om_1, _, _ = seq_1(spin, interp_only=interp_only)
        om_2, _, _ = seq_2(spin, interp_only=interp_only)
        return np.abs(om_1 - om_2)

    if np.ndim(a) == 0:
        return _gap(float(a))
    return np.array([_gap(float(spin)) for spin in np.asarray(a)])


def find_min_gap(mode_1, mode_2, a_min=0., a_max=0.999, num=200,
                 interp_only=True):
    """Locate the minimum frequency gap between two QNM spin sequences.

    A coarse scan over ``num`` spins in [a_min, a_max] is refined with a
    golden-section search around the best grid point.

    Parameters
    ----------
    mode_1: tuple
      Mode labels (s, l, m, n) of the first mode.

    mode_2: tuple
      Mode labels (s, l, m, n) of the second mode.

    a_min: float, optional [default: 0.]
    a_max: float, optional [default: 0.999]
      Spin window to search.

    num: int, optional [default: 200]
      Number of points in the coarse scan.

    Returns
    -------
    (float, float)
      (a_star, gap_star): spin of closest approach and the gap there.
    """

    spins = np.linspace(a_min, a_max, num)
    gaps = mode_gap(spins, mode_1, mode_2, interp_only=interp_only)
    i = int(np.argmin(gaps))
    lo = spins[max(i - 1, 0)]
    hi = spins[min(i + 1, num - 1)]

    invphi = (np.sqrt(5.) - 1.) / 2.
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    fc = mode_gap(c, mode_1, mode_2, interp_only=interp_only)
    fd = mode_gap(d, mode_1, mode_2, interp_only=interp_only)
    while hi - lo > 1e-6:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = mode_gap(c, mode_1, mode_2, interp_only=interp_only)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = mode_gap(d, mode_1, mode_2, interp_only=interp_only)
    a_star = 0.5 * (lo + hi)
    return a_star, float(mode_gap(a_star, mode_1, mode_2,
                                  interp_only=interp_only))
