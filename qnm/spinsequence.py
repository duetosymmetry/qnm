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
    a_max: float [default: .99]
      Maximum dimensionless spin of black hole for the sequence,
      0 <= a_max < 1.

    delta_a: float [default: 0.005]
      Step size in a for following the sequence from a=0 to a_max

    delta_a_min: float [default: 1.e-5]
      Minimum step size in a.

    delta_a_max: float [default: 4.e-3]
      Maximum step size in a.

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
        self.a_max       = kwargs.get('a_max',       0.99)
        # TODO Maybe change this to delta_a0
        self.delta_a     = kwargs.get('delta_a',     0.005)
        self.delta_a_min = kwargs.get('delta_a_min', 1.e-5)
        self.delta_a_max = kwargs.get('delta_a_max', 4.e-3)
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

        while _a <= self.a_max:

            self.solver.set_params(a=_a, A_closest_to=A0,
                                   omega_guess=omega_guess)

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

            # TODO Behavior based on these numbers? Should we continue
            # even if cf_err is larger than the desired error tol?
            cf_err, n_frac = self.solver.get_cf_err()

            # Done with this value of a
            self.a.append(_a)
            self.omega.append(result)
            self.A.append(self.solver.A)
            self.C.append(self.solver.C)
            self.cf_err.append(cf_err)
            self.n_frac.append(n_frac)

            # We always try to get the a_max value. If that's the
            # value we just did, break out of the loop by hand
            if (_a == self.a_max):
                break

            _a, omega_guess, A0 = self._propose_next_a_om_A()

            i = i+1
            # Go to the next iteration of the _a loop

        logging.info("s={}, l={}, m={}, n={} completed with {} points".format(
            self.s, self.l, self.m, self.n, len(self.a)))

        # Done extending the spin sequence!

        self.build_interps()

    def _propose_next_a_om_A(self):
        """This is an internal function that's used by
        do_find_sequence to compute starting values for the next
        step along the spin sequence."""

        # For the next value of a, start with a guess based on
        # the previously-computed values. When we have two or more
        # values, we can do a quadratic fit. Otherwise just start
        # at the same value.
        _a = self.a[-1]

        if (len(self.a) < 3):
            omega_guess = self.omega[-1]
            A0          = self.A[-1]

            _a = _a + self.delta_a
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

            if (True or (len(self.a) > np.Inf)): # Only do the curvature estimate after a while
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

        return _a, omega_guess, A0

    def build_interps(self):
        """Build interpolating functions for omega(a) and A(a).

        This is automatically called at the end of :meth:`do_find_sequence`.
        """

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

    def __call__(self, a, store=False):
        """Solve for omega, A, and C[] at a given spin a.

        This uses the interpolants, based on the solved sequence, for
        initial guesses of omega(a) and A(a).

        Parameters
        ----------
        a: float
          Value of spin, 0 <= a < 1.

        store: bool, optional [default: False]
          Whether or not to save newly solved data in sequence.
          Warning, this can produce a slowdown if a lot of data
          needs to be moved.

        Returns
        -------
        complex, complex, complex ndarray
          The first element of the tuple is omega. The second element
          of the tuple is A. The third element of the tuple is the
          array of complex spherical-spheroidal decomposition
          coefficients. For documentation on the format of the
          spherical-spheroidal decomposition coefficient array, see
          :mod:`qnm.angular` or
          :func:`qnm.angular.C_and_sep_const_closest`.
        """

        # TODO Make sure that interpolants have been built

        # TODO validate input, 0 <= a < 1.
        # TODO if a > a_max then extend
        # TODO take parameter of whether to solve at guess or not


        # If this was a previously computed value, just return the
        # earlier results
        if (a in self.a):
            a_ind = self.a.index(a)

            return self.omega[a_ind], self.A[a_ind], self.C[a_ind]

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

        # If we got here, then this new value of a was not already in
        # the list
        if store:
            # Where do we want to insert? Before the first point with
            # a larger spin
            try:
                insert_ind = next(i for i, _a in
                                  enumerate( self.a ) if _a > a)
            except StopIteration:
                insert_ind = len(self.a)

            self.a.insert(insert_ind, a)
            self.omega.insert(insert_ind, result)
            self.A.insert(insert_ind, self.solver.A)
            self.C.insert(insert_ind, self.solver.C)
            self.cf_err.insert(insert_ind, cf_err)
            self.n_frac.insert(insert_ind, n_frac)

        return result, self.solver.A, self.solver.C

    def __repr__(self):
    # "The goal of __str__ is to be readable; the goal of __repr__ is to be unambiguous." --- stackoverflow
        from textwrap import dedent

        rep = """<{} with s={}, l={}, m={}, n={},
             l_max={},
             tol={},
             Nr={}, Nr_min={}, Nr_max={},
             with values at
             a=[{}, ... <{}> ..., {}]>"""
        rep = rep.format(type(self).__name__,
                         str(self.s), str(self.l),
                         str(self.m), str(self.n),
                         str(self.l_max),
                         str(self.tol),
                         str(self.Nr), str(self.Nr_min), str(self.Nr_max),
                         str(self.a[0]), str(len(self.a)-2), str(self.a[-1]))

        return dedent(rep)
