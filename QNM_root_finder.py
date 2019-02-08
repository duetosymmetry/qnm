from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import optimize

import angular
import radial

# TODO some documentation here, better documentation throughout

class nearby_root_finder(object):

    # TODO: This is too many positional args, use kwargs
    def __init__(self, a=0., s=-2, m=2, A_closest_to=4.+0.j, l_max=20,
                 omega_guess=1.-1.j, tol=1e-10, n_inv=0, Nr=300,
                 r_N=0.j):
        """Object to find and store results from simultaneous roots of
        radial and angular QNM equations, following the
        Leaver and Cook-Zalutskiy approach.

        Parameters
        ==========
        a: float [default: 0.]
          Dimensionless spin of black hole, 0 <= a < 1.

        s: int [default: 2]
          Spin of field of interest

        m: int [default: 2]
          Azimuthal number of mode of interest

        A_closest_to: complex [default: 4.+0.j]
          Complex value close to desired separation constant. This is
          intended for tracking the l-number of a branch starting from
          the analytically-known value at a=0

        l_max: int [default: 20]
          Maximum value of l to include in the spherical-spheroidal
          matrix for finding separation constant and mixing
          coefficients. Must be sufficiently larger than l of interest
          that angular spectral method can converge. The number of
          l's needed for convergence depends on a.

        omega_guess: complex [default: 1.-1.j]
          Initial guess of omega for root-finding

        tol: float [default: 1e-10]
          Tolerance for root-finding

        n_inv: int [default: 0]
          Inversion number of radial infinite continued fraction,
          which selects overtone number of interest

        Nr: int [default: 300]
          Truncation number of radial infinite continued
          fraction. Must be sufficiently large for convergence.

        r_N: complex [default: 0.j]
          Seed value taken for truncation of infinite continued
          fraction.

        """

        # TODO: Check that values make sense
        self.a           = a
        self.s           = s
        self.m           = m
        self.A0          = A_closest_to
        self.l_max       = l_max
        self.omega_guess = omega_guess
        self.tol         = tol
        self.n_inv       = n_inv
        self.Nr          = Nr
        self.r_N         = r_N

        # Where the results will go
        self.opt_res = None

        self.omega = None
        self.A     = None
        self.C     = []

    def __call__(self, x, tol):
        """Internal function for usage with optimize.root, for an
        instance of this class to act like a function for
        root-finding. optimize.root only works with reals so we pack
        and unpack complexes into float[2]
        """

        omega = x[0] + 1.j*x[1]
        # oblateness parameter
        c     = self.a * omega
        # Separation constant at this a*omega
        A     = angular.sep_const_closest(self.A0, self.s, c, self.m,
                                          self.l_max)

        # We are trying to find a root of this function:
        inv_err = radial.Leaver_Cf_trunc_inversion(omega, self.a,
                                                   self.s, self.m, A,
                                                   self.n_inv,
                                                   self.Nr, self.r_N)

        return [np.real(inv_err), np.imag(inv_err)]

    def clear_results(self):
        """ TODO Documentation """
        self.omega = None
        self.A     = None
        self.C     = None
    
    def do_solve(self):
        """ TODO Documentation """

        self.opt_res = optimize.root(self,
                                     [np.real(self.omega_guess), np.imag(self.omega_guess)],
                                     self.tol)

        if (not self.opt_res.success):
            self.clear_results()
            return None

        self.omega = self.opt_res.x[0] + 1.j*self.opt_res.x[1]
        c = self.a * self.omega
        self.A, self.C = angular.C_and_sep_const_closest(self.A0,
                                                         self.s, c,
                                                         self.m, self.l_max)

        return self.omega
