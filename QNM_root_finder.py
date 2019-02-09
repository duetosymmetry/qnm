from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from scipy import optimize

import angular
import radial
from Schw_QNM_expans import Schw_QNM_estimate

# TODO some documentation here, better documentation throughout

class nearby_root_finder(object):

    def __init__(self, *args, **kwargs):
        """Object to find and store results from simultaneous roots of
        radial and angular QNM equations, following the
        Leaver and Cook-Zalutskiy approach.

        Keyword arguments
        ==========
        a: float [default: 0.]
          Dimensionless spin of black hole, 0 <= a < 1.

        s: int [default: 2]
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

        tol: float [default: 1e-10]
          Tolerance for root-finding

        n_inv: int [default: 0]
          Inversion number of radial infinite continued fraction,
          which selects overtone number of interest

        Nr: int [default: 300]
          Truncation number of radial infinite continued
          fraction. Must be sufficiently large for convergence.

        r_N: complex [default: 1.]
          Seed value taken for truncation of infinite continued
          fraction.

        """

        # Set defaults before using values in kwargs
        self.a           = 0.
        self.s           = -2
        self.m           = 2
        self.A0          = 4.+0.j
        self.l_max       = 20
        self.omega_guess = .5-.5j
        self.tol         = 1e-10
        self.n_inv       = 0
        self.Nr          = 300
        self.r_N         = 1.

        self.set_params(**kwargs)

    def set_params(self, *args, **kwargs):
        """Set the parameters for root finding. Parameters are
        described in the class documentation. Finally calls
        clear_results().
        """

        self.a           = kwargs.get('a',            self.a)
        self.s           = kwargs.get('s',            self.s)
        self.m           = kwargs.get('m',            self.m)
        self.A0          = kwargs.get('A_closest_to', self.A0)
        self.l_max       = kwargs.get('l_max',        self.l_max)
        self.omega_guess = kwargs.get('omega_guess',  self.omega_guess)
        self.tol         = kwargs.get('tol',          self.tol)
        self.n_inv       = kwargs.get('n_inv',        self.n_inv)
        self.Nr          = kwargs.get('Nr',           self.Nr)
        self.r_N         = kwargs.get('r_N',          self.r_N)

        # TODO: Check that values make sense

        self.clear_results()

    def clear_results(self):
        """ TODO Documentation """

        self.solved  = False
        self.opt_res = None

        self.omega = None
        self.A     = None
        self.C     = None


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

    def do_solve(self):
        """ TODO Documentation """

        self.opt_res = optimize.root(self,
                                     [np.real(self.omega_guess), np.imag(self.omega_guess)],
                                     self.tol)

        if (not self.opt_res.success):
            self.clear_results()
            return None

        self.solved = True

        self.omega = self.opt_res.x[0] + 1.j*self.opt_res.x[1]
        c = self.a * self.omega
        self.A, self.C = angular.C_and_sep_const_closest(self.A0,
                                                         self.s, c,
                                                         self.m, self.l_max)

        return self.omega

    def estimate_cf_err(self):
        """ TODO Documentation """

        if not self.solved:
            raise Exception("Can only approximate continued fraction "
                            "error after successfully solving")

        err1 = radial.Leaver_Cf_trunc_inversion(self.omega, self.a,
                                                self.s, self.m, self.A,
                                                self.n_inv,
                                                self.Nr, self.r_N)
        err2 = radial.Leaver_Cf_trunc_inversion(self.omega, self.a,
                                                self.s, self.m, self.A,
                                                self.n_inv,
                                                self.Nr + 1,
                                                self.r_N)
        cf_err = np.abs(err1 - err2)

        return cf_err

class QNM_seq_root_finder(object):

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

        delta_a: float [default: 0.01]
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

        omega_guess: complex [default: from Schw_QNM_estimate]
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
        self.delta_a     = kwargs.get('delta_a',     0.01)
        self.s           = kwargs.get('s',           -2)
        self.m           = kwargs.get('m',           2)
        self.l           = kwargs.get('l',           2)
        self.l_max       = kwargs.get('l_max',       20)
        self.tol         = kwargs.get('tol',         1e-10)
        self.n           = kwargs.get('n',           0)
        self.omega_guess = kwargs.get('omega_guess',
                                      Schw_QNM_estimate(self.s,
                                                        self.n,
                                                        self.l))
        self.Nr          = kwargs.get('Nr',          300)
        self.Nr_min      = self.Nr
        self.Nr_max      = 3000    # TODO Get rid of magic number
        self.r_N         = kwargs.get('r_N',         0.j)

        # TODO check that values make sense!!!

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

                # Check if the continued fraction has less error than
                # the desired tolerance
                cf_err = self.solver.estimate_cf_err()

                if ((cf_err < self.tol) or (self.Nr >= self.Nr_max)):
                    # Converged or can't increase
                    cf_conv = True # Move on to next value of a
                    self.omega[i] = result
                    self.A[i]     = self.solver.A
                    self.C[i]     = self.solver.C
                    self.cf_err[i]= cf_err
                    # Can we relax?
                    if ((cf_err / self.tol < 1e-2) and (self.Nr > self.Nr_min)):
                        self.Nr = self.Nr - 50
                        logging.info("Converged for a={}, required {}"
                                     " func evals".format(_a,
                                                          self.solver.opt_res.nfev))
                        logging.info("Going to relax Nr to {}".format(self.Nr))
                        self.solver.set_params(Nr=self.Nr)
                else:
                    # Need to add more terms to cont. frac. approx
                    # TODO Don't make this a magic number
                    self.Nr = self.Nr + 100
                    if (self.Nr >= self.Nr_max):
                        logging.warning("Nr={} hit Nr_max ".format(self.Nr))
                    # Should we warn?
                    # Should we have a max?
                    logging.info("Increasing Nr to {}".format(self.Nr))
                    self.solver.set_params(Nr=self.Nr,
                                           omega_guess=result,
                                           A_closest_to=self.solver.A)
                    # Now try again, because cf_conv is still False


            # Every time through the loop, use previous result
            omega_guess = self.omega[i]
            A0          = self.A[i]

        logging.info("n={}, l={}, started from guess omega={}, "
                     "found omega={}".format(self.n, self.l,
                                             self.omega_guess, self.omega[0]))
