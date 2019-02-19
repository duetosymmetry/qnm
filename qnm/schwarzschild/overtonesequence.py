""" Follow a Schwarzschild QNM sequence (s,l) from n=0 upwards.

The class :class:`SchwOvertoneSeq` makes it possible to find
successive overtones (n's) of a QNM labeled by (s,l), in Schwarzschild
(a=0). See its documentation for more details.
"""

from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from scipy import optimize, interpolate

from ..angular import l_min, swsphericalh_A
from ..nearby import NearbyRootFinder
from .approx import dolan_ottewill_expansion

# TODO some documentation here, better documentation throughout

class SchwOvertoneSeq(object):
    """Object to follow a sequence of Schwarzschild overtones,
    starting from n=0.  First two overtone seeds come from
    approx.dolan_ottewill_expansion, and afterwards linear
    extrapolation on the solutions is used to seed the root
    finding for higher values of n. Uses
    :class:`qnm.nearby.NearbyRootFinder` to actually perform the
    root-finding.

    Attributes
    ----------
    A: float
      Value of the angular separation constant.

    n: array of int
      Overtone numbers.

    omega: np.array of complex
      The QNM frequencies along the overtone sequence, element i is
      overtone i.

    cf_err: np.array of float
      Estimate of continued fraction truncation error in solving for
      QNM frequency.

    n_frac: np.array of int
      Truncation number of continued fraction.

    solver: NearbyRootFinder
      Instance of :class:`qnm.nearby.NearbyRootFinder` that is used to
      find the QNMs.

    Parameters
    ----------
    n_max: int [default: 12]
      Maximum overtone number to search for (must be positive).

    s: int [default: 2]
      Spin weight of field of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    l: int [default: 2]
      The multipole number of a sequence starting from the
      analytically-known value at a=0 .

    tol: float [default: 1e-10]
      Tolerance for root-finding.

    Nr: int [default: 300]
      Truncation number of radial infinite continued
      fraction. Must be sufficiently large for convergence.

    r_N: complex [default: 0.j]
      Seed value taken for truncation of infinite continued
      fraction.

    Examples
    --------
    Suppose you want the n=5 overtone for (s=-1, l=3):

    >>> from qnm.schwarzschild.overtonesequence import SchwOvertoneSeq
    >>> seq = SchwOvertoneSeq(s=-1, l=3, n_max=5)
    >>> seq.find_sequence()
    >>> print(seq.omega[5])
    (0.5039017454081958-1.1703558890487786j)

    Later, you want to go out to n=8:

    >>> seq.extend(n_max=8)
    >>> print(seq.omega[8])
    (0.4227909293895908-1.9136575597714864j)

    """

    def __init__(self, *args, **kwargs):

        # Read args
        self.n_max       = kwargs.get('n_max',       12)
        self.s           = kwargs.get('s',           -2)
        self.l           = kwargs.get('l',           2)
        self.tol         = kwargs.get('tol',         1e-10)
        self.Nr          = kwargs.get('Nr',          300)
        self.Nr_min      = self.Nr
        self.Nr_max      = kwargs.get('Nr_max',      6000)
        self.r_N         = kwargs.get('r_N',         0.j)

        # TODO check that values make sense!!!
        assert self.l >= l_min(self.s, 0), ("l={} must be >= "
                                            "l_min={}".format(
                                                self.l,
                                                l_min(self.s, 0)))

        # We know the Schwarzschild separation constant analytically
        self.A = swsphericalh_A(self.s, self.l, 0)

        # Create array of n's and omega's
        self.n      = []
        self.omega  = np.array([], dtype=complex)
        self.cf_err = np.array([])
        self.n_frac  = np.array([])

        # We need and instance of root finder
        self.solver = NearbyRootFinder(s=self.s, m=0,
                                       l_max=self.l + 1,
                                       a=0.,
                                       A_closest_to=self.A,
                                       tol=self.tol,
                                       n_inv=0, Nr=self.Nr,
                                       Nr_max=self.Nr_max,
                                       r_N=self.r_N)

    def find_sequence(self):
        """ Alias for :meth:`extend()` """

        self.extend()

    def extend(self, n_max=None):
        """ Extend the current overtone sequence to a greater n_max.

        Parameters
        ----------
        n_max: int, optional [default: None]
          If is None, use the value of self.n_max .
          If given, set self.n_max to this new value and proceed.

        Raises
        ------
        optimize.nonlin.NoConvergence
          If a root can't be found within a few [TODO make param?]
          attempted inversion numbers.

        """

        if (n_max is not None):
            self.n_max = n_max

        while (len(self.omega) <= self.n_max):

            n = len(self.omega)

            self.solver.clear_results()
            self.solver.set_params(n_inv=n)

            if (n < 2):
                omega_guess = dolan_ottewill_expansion(self.s, self.l, n)
            else:
                # Linearly extrapolate from the last two
                om_m_1 = self.omega[-1]
                om_m_2 = self.omega[-2]

                om_diff = om_m_1 - om_m_2

                # Linearly interpolate
                omega_guess = om_m_1 + om_diff

                if (n > 5):
                    # Check if this difference is greater than the typical spacing
                    typ_sp = np.mean(np.abs(np.diff(self.omega[:-1])))

                    if (np.abs(om_diff) > 2. * typ_sp):
                        # It's likely we skipped an overtone.
                        # Average and go back one
                        omega_guess = (om_m_1 + om_m_2)/2.
                        self.solver.set_params(n_inv=n-1)
                        logging.info("Potentially skipped an overtone "
                                     "in the series, trying to go back")

            self.solver.set_params(omega_guess=omega_guess)
            # Try to reject previously-found poles
            self.solver.set_poles(self.omega)

            # Flag: is the continued fraction expansion converging?
            cf_conv = False

            while not cf_conv:

                result = self.solver.do_solve()

                if (result is None):
                    # Potentially try the next inversion number
                    cur_inv_n = self.solver.n_inv
                    cur_inv_n = cur_inv_n + 1
                    self.solver.set_params(n_inv=cur_inv_n)

                    if (np.abs(cur_inv_n - n) > 2):
                        # Got too far away, give up
                        raise optimize.nonlin.NoConvergence('Failed to find '
                                                            'QNM in sequence '
                                                            'at n={}'.format(n))
                    else:
                        logging.info("Trying inversion number {}".format(cur_inv_n))
                        # Try again
                        continue

                # Ensure we start on the "positive frequency"
                # sequence.  This only works for Schwarzscdhild because
                # there the separation constant is real.
                if (np.real(result) < 0):
                    result = -np.conjugate(result)

                cf_err, n_frac = self.solver.get_cf_err()

                # TODO ACTUALLY DO SOMETHING WITH THESE NUMBERS
                cf_conv = True

                if cf_conv:
                    # Done with this value of n

                    logging.info("s={}, l={}, found what I think is "
                                 "n={}, omega={}".format(self.s,
                                                         self.l, n, result))

                    self.n.append(n)

                    self.omega  = np.append(self.omega,  result)
                    self.cf_err = np.append(self.cf_err, cf_err)
                    self.n_frac = np.append(self.n_frac, n_frac)

                    # Make sure we sort properly!
                    ind_sort = np.argsort(-np.imag(self.omega))
                    self.omega  = self.omega[ind_sort]
                    self.cf_err = self.cf_err[ind_sort]
                    self.n_frac = self.n_frac[ind_sort]

                else:
                    # For the next attempt, try starting where we
                    # ended up
                    self.solver.set_params(omega_guess=result)
                    # Now try again, because cf_conv is still False
