""" Follow a QNM labeled by (s,l,m,n) as spin varies from a=0 upwards.

TODO Documentation.
"""


from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from scipy import optimize, interpolate

from .angular import l_min, swsphericalh_A
from .nearby import NearbyRootFinder

from .schwarzschild.approx import Schw_QNM_estimate
from .schwarzschild.tabulated import QNMDict

# TODO some documentation here, better documentation throughout

class KerrSpinSeq(object):
    """Object to follow a QNM up a sequence in a, starting from
    a=0. Values for omega and the separation constant from one
    value of a are used to seed the root finding for the next
    value of a, to maintain continuity in a when separation
    constant order can change. Uses NearbyRootFinder to actually
    perform the root-finding.

    Parameters
    ----------
    a_max: float [default: .9]
      Maximum dimensionless spin of black hole for the sequence,
      0 <= a_max < 1.

    delta_a: float [default: 0.005]
      Step size in a for following the sequence from a=0 to a_max

    s: int [default: 2]
      Spin of field of interest

    m: int [default: 2]
      Azimuthal number of mode of interest

    l: int [default: 2]
      The l-number of a sequence starting from the
      analytically-known value at a=0

    l_max: int [default: 20]
      Maximum value of l to include in the spherical-spheroidal
      matrix for finding separation constant and mixing
      coefficients. Must be sufficiently larger than l of interest
      that angular spectral method can converge. The number of
      l's needed for convergence depends on a.

    omega_guess: complex [default: from schwarzschild.QNMDict]
      Initial guess of omega for root-finding

    tol: float [default: 1e-10]
      Tolerance for root-finding

    n: int [default: 0]
      Overtone number of interest (sets the inversion number for
      infinite continued fraction in Leaver's method)

    Nr: int [default: 300]
      Truncation number of radial infinite continued
      fraction. Must be sufficiently large for convergence.

    Nr_min: int [default: Nr]
      Minimum number of terms for evaluating continued fraction.

    Nr_max: int [default: 4000]
      Maximum number of terms for evaluating continued fraction.

    r_N: complex [default: 0.j]
      Seed value taken for truncation of infinite continued
      fraction. UNUSED, REMOVE

    """

    def __init__(self, *args, **kwargs):

        # Read args
        self.a_max       = kwargs.get('a_max',       0.9)
        # TODO Maybe change this to delta_a0
        self.delta_a     = kwargs.get('delta_a',     0.005)
        self.delta_a_min = 1.e-5 # TODO get rid of magic number
        self.delta_a_max = 4.e-3 # TODO get rid of magic number
        self.s           = kwargs.get('s',           -2)
        self.m           = kwargs.get('m',           2)
        self.l           = kwargs.get('l',           2)
        self.l_max       = kwargs.get('l_max',       20)
        self.tol         = kwargs.get('tol',         1e-10)
        self.n           = kwargs.get('n',           0)

        if ('omega_guess' in kwargs.keys()):
            self.omega_guess = kwargs.get('omega_guess')
        else:
            qnm_dict = QNMDict()
            self.omega_guess = qnm_dict(self.s, self.l, self.n)[0]

        self.Nr          = kwargs.get('Nr',          300)
        self.Nr_min      = self.Nr
        self.Nr_max      = kwargs.get('Nr_max',      4000)
        self.r_N         = kwargs.get('r_N',         0.j)

        # TODO check that values make sense!!!
        assert self.a_max < 1., ("a_max={} must be < 1.".format(self.a_max))
        assert self.l >= l_min(self.s, self.m), ("l={} must be "
                                                 ">= l_min={}".format(
                                                     self.l,
                                                     l_min(self.s, self.m)))

        # Create array of a's, omega's, and A's
        self.a     = []
        self.omega = []
        self.cf_err= []
        self.n_frac= []
        self.A     = []
        self.C     = []

        self.delta_a_prop = []

        self._interp_o_r = None
        self._interp_o_i = None
        self._interp_A_r = None
        self._interp_A_i = None

        # We need and instance of root finder
        self.solver = NearbyRootFinder(s=self.s, m=self.m,
                                       l_max=self.l_max,
                                       tol=self.tol,
                                       n_inv=self.n, Nr=self.Nr,
                                       Nr_min=self.Nr_min,
                                       Nr_max=self.Nr_max,
                                       r_N=self.r_N)

    # Change this to *extending* the sequence so this can be reused
    def do_find_sequence(self):
        """ TODO Document """

        logging.info("l={}, m={}, n={} starting".format(
            self.l, self.m, self.n))
        
        i  = 0  # TODO Allow to start at other values
        _a = 0. # TODO Allow to start at other values

        # Initializing the sequence, start with guesses
        A0 = swsphericalh_A(self.s, self.l, self.m)
        omega_guess = self.omega_guess

        _delta_a = self.delta_a

        while _a <= self.a_max:

            self.solver.set_params(a=_a, A_closest_to=A0,
                                   omega_guess=omega_guess)

            # Flag: is the continued fraction expansion converging?
            cf_conv = False

            while not cf_conv:

                result = self.solver.do_solve()

                if (result is None):
                    raise optimize.nonlin.NoConvergence('Failed to find '
                                                        'QNM in sequence '
                                                        'at a={}'.format(_a))

                # TODO This probably doesn't belong here
                # Ensure we start on the "positive frequency"
                # sequence.  This only works for i==0 (a=0.) because
                # there the separation constant is real.
                if ((i == 0) and (np.real(result) < 0)):
                    result = -np.conjugate(result)


                cf_err, n_frac = self.solver.get_cf_err()

                # TODO ACTUALLY DO SOMETHING WITH THESE NUMBERS
                cf_conv = True

                if cf_conv:
                    # Done with this value of a
                    self.a.append(_a)
                    self.omega.append(result)
                    self.A.append(self.solver.A)
                    self.C.append(self.solver.C)
                    self.cf_err.append(cf_err)
                    self.n_frac.append(n_frac)
                else:
                    # For the next attempt, try starting where we
                    # ended up
                    self.solver.set_params(omega_guess=result,
                                           A_closest_to=self.solver.A)
                    # Now try again, because cf_conv is still False

            # We always try to get the a_max value. If that's the
            # value we just did, break out of the loop by hand
            if (_a == self.a_max):
                break

            # For the next value of a, start with a guess based on
            # the previously-computed values. When we have two or more
            # values, we can do a quadratic fit. Otherwise just start
            # at the same value.
            if (i < 2):
                omega_guess = self.omega[-1]
                A0          = self.A[-1]

                _a = _a + _delta_a
            else:

                # Build interpolants and allow extrapolation
                # Sadly, UnivariateSpline does not work on complex data
                interp_o_r = interpolate.UnivariateSpline(
                    self.a[-3:], np.real(self.omega[-3:]),
                    s=0, # No smoothing!
                    k=2, ext=0)

                interp_o_i = interpolate.UnivariateSpline(
                    self.a[-3:], np.imag(self.omega[-3:]),
                    s=0, # No smoothing!
                    k=2, ext=0)

                interp_A_r = interpolate.UnivariateSpline(
                    self.a[-3:], np.real(self.A[-3:]),
                    s=0, # No smoothing!
                    k=2, ext=0)

                interp_A_i = interpolate.UnivariateSpline(
                    self.a[-3:], np.imag(self.A[-3:]),
                    s=0, # No smoothing!
                    k=2, ext=0)

                if (True or (i > np.Inf)): # Only do the curvature estimate after a while
                    # Their second derivatives
                    d2_o_r = interp_o_r.derivative(2)
                    d2_o_i = interp_o_i.derivative(2)
                    d2_A_r = interp_A_r.derivative(2)
                    d2_A_i = interp_A_i.derivative(2)

                    # Estimate the a-curvature of the omega and A functions:
                    d2_o = np.abs(d2_o_r(_a) + 1.j*d2_o_i(_a))
                    d2_A = np.abs(d2_A_r(_a) + 1.j*d2_A_i(_a))

                    # Get the larger of the two a-curvatures
                    d2 = np.max([d2_o, d2_A])

                    # This combination has units of a. The numerator
                    # is an empirical fudge factor
                    _delta_a = 0.05/np.sqrt(d2)

                self.delta_a_prop.append(_delta_a)

                # Make sure it's between our min and max allowed step size
                _delta_a = np.max([self.delta_a_min, _delta_a])
                _delta_a = np.min([self.delta_a_max, _delta_a])

                _a = _a + _delta_a

                # Make sure we get the end point
                if (_a > self.a_max):
                    _a = self.a_max

                omega_guess = interp_o_r(_a) + 1.j*interp_o_i(_a)
                A0          = interp_A_r(_a) + 1.j*interp_A_i(_a)

            i = i+1
            # Go to the next iteration of the loop

        logging.info("l={}, m={}, n={} completed with {} points".format(
            self.l, self.m, self.n, len(self.a)))

        self.build_interps()

    def build_interps(self):
        """ TODO document """

        # TODO do we want to allow extrapolation?

        k=3 # cubic

        # Sadly, UnivariateSpline does not work on complex data
        self._interp_o_r = interpolate.UnivariateSpline(
            self.a, np.real(self.omega),
            s=0, # No smoothing!
            k=k, ext=0)

        self._interp_o_i = interpolate.UnivariateSpline(
            self.a, np.imag(self.omega),
            s=0, # No smoothing!
            k=k, ext=0)

        self._interp_A_r = interpolate.UnivariateSpline(
            self.a, np.real(self.A),
            s=0, # No smoothing!
            k=k, ext=0)

        self._interp_A_i = interpolate.UnivariateSpline(
            self.a, np.imag(self.A),
            s=0, # No smoothing!
            k=k, ext=0)

    def __call__(self, a):
        """ TODO document """

        # TODO validate input, 0 <= a < 1.
        # TODO if a > a_max then extend
        # TODO take parameter of whether to solve at guess or not

        o_r = self._interp_o_r(a)
        o_i = self._interp_o_i(a)

        A_r = self._interp_A_r(a)
        A_i = self._interp_A_i(a)

        omega_guess = complex(o_r, o_i)
        A_guess     = complex(A_r, A_i)

        self.solver.set_params(a=a, omega_guess=omega_guess,
                               A_closest_to=A_guess)

        result = self.solver.do_solve()

        if (result is None):
            raise optimize.nonlin.NoConvergence('Failed to find '
                                                'QNM in sequence '
                                                'at a={}'.format(a))

        cf_err, n_frac = self.solver.get_cf_err()

        # Do we want to insert these numbers into the arrays?

        return result, self.solver.A, self.solver.C
