""" Find a nearby root of the coupled radial/angular Teukolsky equations.

TODO Documentation.
"""

from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from scipy import optimize

from .angular import sep_const_closest, C_and_sep_const_closest
from . import radial

# TODO some documentation here, better documentation throughout

class NearbyRootFinder(object):
    """Object to find and store results from simultaneous roots of
    radial and angular QNM equations, following the
    Leaver and Cook-Zalutskiy approach.

    Parameters
    ----------
    a: float [default: 0.]
      Dimensionless spin of black hole, 0 <= a < 1.

    s: int [default: -2]
      Spin of field of interest

    m: int [default: 2]
      Azimuthal number of mode of interest

    A_closest_to: complex [default: 4.+0.j]
      Complex value close to desired separation constant. This is
      intended for tracking the l-number of a sequence starting
      from the analytically-known value at a=0

    l_max: int [default: 20]
      Maximum value of l to include in the spherical-spheroidal
      matrix for finding separation constant and mixing
      coefficients. Must be sufficiently larger than l of interest
      that angular spectral method can converge. The number of
      l's needed for convergence depends on a.

    omega_guess: complex [default: .5-.5j]
      Initial guess of omega for root-finding

    tol: float [default: sqrt(double epsilon)]
      Tolerance for root-finding omega

    cf_tol: float [defailt: 1e-10]
      Tolerance for continued fraction calculation

    n_inv: int [default: 0]
      Inversion number of radial infinite continued fraction,
      which selects overtone number of interest

    Nr: int [default: 300]
      Truncation number of radial infinite continued
      fraction. Must be sufficiently large for convergence.

    Nr_min: int [default: 300]
      Floor for Nr (for dynamic control of Nr)

    Nr_max: int [default: 4000]
      Ceiling for Nr (for dynamic control of Nr)

    r_N: complex [default: 1.]
      Seed value taken for truncation of infinite continued
      fraction. UNUSED, REMOVE

    """

    def __init__(self, *args, **kwargs):

        # Set defaults before using values in kwargs
        self.a           = 0.
        self.s           = -2
        self.m           = 2
        self.A0          = 4.+0.j
        self.l_max       = 20
        self.omega_guess = .5-.5j
        self.tol         = np.sqrt(np.finfo(float).eps)
        self.cf_tol      = 1e-10
        self.n_inv       = 0
        self.Nr          = 300
        self.Nr_min      = 300
        self.Nr_max      = 4000
        self.r_N         = 1.

        self.set_params(**kwargs)

    def set_params(self, *args, **kwargs):
        """Set the parameters for root finding. Parameters are
        described in the class documentation. Finally calls
        :meth:`clear_results`.
        """

        # TODO This violates DRY, do better.
        self.a           = kwargs.get('a',            self.a)
        self.s           = kwargs.get('s',            self.s)
        self.m           = kwargs.get('m',            self.m)
        self.A0          = kwargs.get('A_closest_to', self.A0)
        self.l_max       = kwargs.get('l_max',        self.l_max)
        self.omega_guess = kwargs.get('omega_guess',  self.omega_guess)
        self.tol         = kwargs.get('tol',          self.tol)
        self.cf_tol      = kwargs.get('cf_tol',       self.cf_tol)
        self.n_inv       = kwargs.get('n_inv',        self.n_inv)
        self.Nr          = kwargs.get('Nr',           self.Nr)
        self.Nr_min      = kwargs.get('Nr_min',       self.Nr_min)
        self.Nr_max      = kwargs.get('Nr_max',       self.Nr_max)
        self.r_N         = kwargs.get('r_N',          self.r_N)

        # Optional pole factors
        self.poles       = np.array([])

        # TODO: Check that values make sense

        self.clear_results()

    def clear_results(self):
        """Clears the stored results from last call of :meth:`do_solve`"""

        self.solved  = False
        self.opt_res = None

        self.omega = None
        self.A     = None
        self.C     = None

        self.cf_err = None
        self.n_frac = None

        self.poles = np.array([])


    def __call__(self, x):
        """Internal function for usage with optimize.root, for an
        instance of this class to act like a function for
        root-finding. optimize.root only works with reals so we pack
        and unpack complexes into float[2]
        """

        omega = x[0] + 1.j*x[1]
        # oblateness parameter
        c     = self.a * omega
        # Separation constant at this a*omega
        A     = sep_const_closest(self.A0, self.s, c, self.m,
                                  self.l_max)

        # We are trying to find a root of this function:
        # inv_err = radial.leaver_cf_trunc_inversion(omega, self.a,
        #                                            self.s, self.m, A,
        #                                            self.n_inv,
        #                                            self.Nr, self.r_N)

        # TODO!
        # Determine the value to use for cf_tol based on
        # the Jacobian, cf_tol = |d cf(\omega)/d\omega| tol.
        inv_err, self.cf_err, self.n_frac = radial.leaver_cf_inv_lentz(omega, self.a,
                                                          self.s, self.m, A,
                                                          self.n_inv, self.cf_tol,
                                                          self.Nr_min, self.Nr_max)
        # logging.info("Lentz terminated with cf_err={}, n_frac={}".format(self.cf_err, self.n_frac))

        # Insert optional poles
        pole_factors   = np.prod(omega - self.poles)
        supp_err = inv_err / pole_factors

        return [np.real(supp_err), np.imag(supp_err)]

    def do_solve(self):
        """Try to find a root of the continued fraction equation,
        using the parameters that have been set in :meth:`set_params`."""

        # For the default (hybr) method, tol sets 'xtol', the
        # tolerance on omega.
        self.opt_res = optimize.root(self,
                                     [np.real(self.omega_guess), np.imag(self.omega_guess)],
                                     method = 'hybr', tol = self.tol)

        if (not self.opt_res.success):
            tmp_opt_res = self.opt_res
            self.clear_results()
            self.opt_res = tmp_opt_res
            return None

        self.solved = True

        self.omega = self.opt_res.x[0] + 1.j*self.opt_res.x[1]
        c = self.a * self.omega
        # As far as I can tell, scipy.linalg.eig already normalizes
        # the eigenvector to unit norm, and the coefficient with the
        # largest norm is real
        self.A, self.C = C_and_sep_const_closest(self.A0,
                                                 self.s, c,
                                                 self.m, self.l_max)

        return self.omega

    def get_cf_err(self):
        """Return the continued fraction error and the number of
        iterations in the last evaluation of the continued fraction.

        Returns
        -------
        cf_err: float

        n_frac: int
        """

        return self.cf_err, self.n_frac

    def set_poles(self, poles=[]):
        """ Set poles to multiply error function.

        Parameters
        ----------
        poles: array_like as complex numbers [default: []]

        """

        self.poles = np.array(poles).astype(complex)
