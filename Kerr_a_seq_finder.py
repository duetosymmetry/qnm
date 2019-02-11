from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from scipy import optimize, interpolate

import angular
import radial
from nearby_root_finder import nearby_root_finder

from Schw_QNM_expans import Schw_QNM_estimate
from Schw_table import Schw_QNM_dict

# TODO some documentation here, better documentation throughout

class Kerr_a_seq_finder(object):

    def __init__(self, *args, **kwargs):
        """Object to follow a QNM up a sequence in a, starting from
        a=0. Values for omega and the separation constant from one
        value of a are used to seed the root finding for the next
        value of a, to maintain continuity in a when separation
        constant order can change. Uses nearby_root_finder to actually
        perform the root-finding.

        Keyword arguments
        ==========
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

        omega_guess: complex [default: from Schw_QNM_dict or Schw_QNM_estimate]
          Initial guess of omega for root-finding

        tol: float [default: 1e-10]
          Tolerance for root-finding

        n: int [default: 0]
          Overtone number of interest (sets the inversion number for
          infinite continued fraction in Leaver's method)

        Nr: int [default: 300]
          Truncation number of radial infinite continued
          fraction. Must be sufficiently large for convergence.

        r_N: complex [default: 0.j]
          Seed value taken for truncation of infinite continued
          fraction.

        """

        # Read args
        self.a_max       = kwargs.get('a_max',       0.9)
        self.delta_a     = kwargs.get('delta_a',     0.005)
        self.s           = kwargs.get('s',           -2)
        self.m           = kwargs.get('m',           2)
        self.l           = kwargs.get('l',           2)
        self.l_max       = kwargs.get('l_max',       20)
        self.tol         = kwargs.get('tol',         1e-10)
        self.n           = kwargs.get('n',           0)

        self.Schw_QNM_dict = Schw_QNM_dict().load_dict()
        if ((self.s, self.l, self.n) in self.Schw_QNM_dict.keys()):
            def_om_guess = self.Schw_QNM_dict[(self.s, self.l, self.n)]
        else:
            def_om_guess = self.Schw_QNM_estimate(self.s, self.n, self.l)

        self.omega_guess = kwargs.get('omega_guess', def_om_guess)

        self.Nr          = kwargs.get('Nr',          300)
        self.Nr_min      = self.Nr
        self.Nr_max      = 3000    # TODO Get rid of magic number
        self.r_N         = kwargs.get('r_N',         0.j)

        # TODO check that values make sense!!!
        assert self.l >= angular.l_min(self.s, self.m), ("l={} must be "
                                                         ">= l_min={}".format(
                                                             self.l,
                                                             angular.l_min(self.s, self.m)))

        # Create array of a's, omega's, and A's
        self.a     = np.arange(0., self.a_max, self.delta_a)
        self.omega = [None] * len(self.a)
        self.cf_err= [None] * len(self.a)
        self.A     = [None] * len(self.a)
        self.C     = [None] * len(self.a)

        # We need and instance of root finder
        self.solver = nearby_root_finder(s=self.s, m=self.m,
                                         l_max=self.l_max,
                                         tol=self.tol,
                                         n_inv=self.n, Nr=self.Nr,
                                         r_N=self.r_N)

    def do_find_sequence(self):

        # Initializing the sequence, start with guesses
        A0 = angular.SWSphericalH_A(self.s, self.l, self.m)
        omega_guess = self.omega_guess

        for i, _a in enumerate(self.a):


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

                # Return value from the auto-adjuster is used to
                # determine if we should try solving again. When the
                # return value is False, that means Nr was increased
                # so we need to try again.
                cf_conv = self.solver.auto_adjust_Nr()

                if cf_conv:
                    # Done with this value of a
                    self.omega[i]  = result
                    self.A[i]      = self.solver.A
                    self.C[i]      = self.solver.C
                    self.cf_err[i] = self.solver.cf_err
                else:
                    # For the next attempt, try starting where we
                    # ended up
                    self.solver.set_params(omega_guess=result,
                                           A_closest_to=self.solver.A)
                    # Now try again, because cf_conv is still False

            # For the next value of a, start with a guess based on
            # the previously-computed values. When we have two or more
            # values, we can do a quadratic fit. Otherwise just start
            # at the same value.
            if (i < 2):
                omega_guess = self.omega[i]
                A0          = self.A[i]
            elif (i+1 < len(self.a)):

                next_a = self.a[i+1]

                # Build an interpolant and allow extrapolation
                interp = interpolate.interp1d(self.a[i-2:i+1],
                                              self.omega[i-2:i+1],
                                              kind='quadratic',
                                              bounds_error=False,
                                              fill_value='extrapolate')
                omega_guess = interp(next_a)

                # Same thing for the separation constant
                interp = interpolate.interp1d(self.a[i-2:i+1],
                                              self.A[i-2:i+1],
                                              kind='quadratic',
                                              bounds_error=False,
                                              fill_value='extrapolate')
                A0 = interp(next_a)



        logging.info("n={}, l={}, started from guess omega={}, "
                     "found omega={}".format(self.n, self.l,
                                             self.omega_guess, self.omega[0]))
