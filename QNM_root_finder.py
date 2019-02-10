from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from scipy import optimize, interpolate

import angular
import radial
from Schw_QNM_expans import Schw_QNM_estimate, Dolan_Ottewill_expansion

# TODO some documentation here, better documentation throughout

class nearby_root_finder(object):

    def __init__(self, *args, **kwargs):
        """Object to find and store results from simultaneous roots of
        radial and angular QNM equations, following the
        Leaver and Cook-Zalutskiy approach.

        Keyword arguments
        =================
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

        Nr_min: int [default: 300]
          Floor for Nr (for dynamic control of Nr)

        Nr_max: int [default: 3000]
          Ceiling for Nr (for dynamic control of Nr)

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
        self.Nr_min      = 300
        self.Nr_max      = 3000
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
        self.Nr_min      = kwargs.get('Nr_min',       self.Nr_min)
        self.Nr_max      = kwargs.get('Nr_max',       self.Nr_max)
        self.r_N         = kwargs.get('r_N',          self.r_N)

        # Optional pole factors
        self.poles       = np.array([])

        # TODO: Check that values make sense

        self.clear_results()

    def clear_results(self):
        """ TODO Documentation """

        self.solved  = False
        self.opt_res = None

        self.omega = None
        self.A     = None
        self.C     = None

        self.cf_err = None

        self.poles = np.array([])


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

        # Insert optional poles
        pole_factors   = np.prod(omega - self.poles)
        supp_err = inv_err / pole_factors

        return [np.real(supp_err), np.imag(supp_err)]

    def do_solve(self):
        """ TODO Documentation """

        self.opt_res = optimize.root(self,
                                     [np.real(self.omega_guess), np.imag(self.omega_guess)],
                                     self.tol)

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
        self.cf_err = np.abs(err1 - err2)

        return self.cf_err

    def auto_adjust_Nr(self):
        """ Try to adjust Nr up or down, depending on whether the
        error estimate in the continued fraction expansion is
        above/below tolerance.

        Returns True  if Nr was relaxed or stayed the same (e.g. hit Nr_max).
        Returns False if Nr was increased.
        """

        self.estimate_cf_err()

        tol_increased = False

        # TODO magic numbers
        if ((self.cf_err < 0.01*self.tol) and (self.Nr > self.Nr_min)):
            # Can relax TODO magic number
            self.Nr = np.max([self.Nr - 50, self.Nr_min])
            logging.info("Relaxing Nr to {}".format(self.Nr))
        elif ((self.cf_err > self.tol) and (self.Nr < self.Nr_max)):
            # Need to add more terms TODO magic number
            self.Nr = np.min([self.Nr + 100, self.Nr_max])
            logging.info("Increasing Nr to {}".format(self.Nr))
            if (self.Nr == self.Nr_max):
                logging.warning("Nr={} has hit Nr_max".format(self.Nr))
            tol_increased = True

        return not tol_increased

    def set_poles(self, poles=[]):
        """ Multiply error function by poles in the complex plane.

        Arguments
        =========
        poles: array-like, as complex numbers [default: []]

        """

        self.poles = np.array(poles).astype(complex)

class Schw_n_seq_finder(object):

    def __init__(self, *args, **kwargs):
        """Object to follow a sequence of Schwarzschild overtones,
        starting from n=0.  First two overtone seeds come from
        Dolan_Ottewill_expansion, and afterwards linear
        extrapolation on the solutions is used to seed the root
        finding for higher values of n. Uses nearby_root_finder to
        actually perform the root-finding.

        Keyword arguments
        ==========
        n_max: int [default: 12]
          Maximum overtone number to search for (must be positive)

        s: int [default: 2]
          Spin of field of interest

        [The m argument is omitted because this is just for Schwarzschild]

        l: int [default: 2]
          The l-number of a sequence starting from the
          analytically-known value at a=0

        l_max: int [default: 20]
          Maximum value of l to include in the spherical-spheroidal
          matrix for finding separation constant and mixing
          coefficients. Must be sufficiently larger than l of interest
          that angular spectral method can converge. The number of
          l's needed for convergence depends on a.

        tol: float [default: 1e-10]
          Tolerance for root-finding

        Nr: int [default: 300]
          Truncation number of radial infinite continued
          fraction. Must be sufficiently large for convergence.

        r_N: complex [default: 0.j]
          Seed value taken for truncation of infinite continued
          fraction.

        """

        # Read args
        self.n_max       = kwargs.get('n_max',       12)
        self.s           = kwargs.get('s',           -2)
        self.l           = kwargs.get('l',           2)
        self.l_max       = kwargs.get('l_max',       20)
        self.tol         = kwargs.get('tol',         1e-10)
        self.Nr          = kwargs.get('Nr',          300)
        self.Nr_min      = self.Nr
        self.Nr_max      = 5000    # TODO Get rid of magic number
        self.r_N         = kwargs.get('r_N',         0.j)

        # TODO check that values make sense!!!
        assert self.l >= angular.l_min(self.s, 0), ("l={} must be >= "
                                                    "l_min={}".format(
                                                        self.l,
                                                        angular.l_min(self.s, 0)))

        # We know the Schwarzschild separation constant analytically
        self.A = angular.SWSphericalH_A(self.s, self.l, 0)

        # Create array of n's and omega's
        self.n      = []
        self.omega  = np.array([], dtype=complex)
        self.cf_err = np.array([])

        # We need and instance of root finder
        self.solver = nearby_root_finder(s=self.s, m=0,
                                         l_max=self.l_max,
                                         a=0.,
                                         A_closest_to=self.A,
                                         tol=self.tol,
                                         n_inv=0, Nr=self.Nr,
                                         Nr_max=self.Nr_max,
                                         r_N=self.r_N)

    def do_find_sequence(self):

        # TODO : Do this as while loop instead of a for loop.
        # Keep track of all the roots found so far and keep them
        # sorted by negative imaginary part.

        while (len(self.omega) < self.n_max):

            n = len(self.omega)

            self.solver.clear_results()
            self.solver.set_params(n_inv=n)

            if (n < 2):
                omega_guess = Dolan_Ottewill_expansion(self.s, n, self.l)
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

                # Return value from the auto-adjuster is used to
                # determine if we should try solving again. When the
                # return value is False, that means Nr was increased
                # so we need to try again.
                cf_conv = self.solver.auto_adjust_Nr()

                if cf_conv:
                    # Done with this value of n

                    logging.info("s={}, l={}, found what I think is "
                                 "n={}, omega={}".format(self.s,
                                                         self.l, n, result))

                    self.n.append(n)

                    self.omega  = np.append(self.omega,  result)
                    self.cf_err = np.append(self.cf_err, self.solver.cf_err)

                    # Make sure we sort properly!
                    ind_sort = np.argsort(-np.imag(self.omega))
                    self.omega  = self.omega[ind_sort]
                    self.cf_err = self.cf_err[ind_sort]

                else:
                    # For the next attempt, try starting where we
                    # ended up
                    self.solver.set_params(omega_guess=result)
                    # Now try again, because cf_conv is still False


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
