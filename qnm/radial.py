""" Solve the radial Teukolsky equation via Leaver's method.

TODO Documentation.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from .contfrac import lentz

# TODO some documentation here, better documentation throughout

def sing_pt_char_exps(omega, a, s, m):
    """ Compute the three characteristic exponents of the singular points
    of the radial Teukolsky equation. We want ingoing at the outer
    horizon and outgoing at infinity. The choice of one of two possible
    characteristic exponents at the inner horizon doesn't affect the minimal
    solution in Leaver's method, so we just pick one.
    """

    root = np.sqrt(1. - a*a)
    r_p, r_m = 1. + root, 1. - root

    sigma_p = (2.*omega*r_p - m*a)/(2.*root)
    sigma_m = (2.*omega*r_m - m*a)/(2.*root)

    zeta = +1.j * omega        # This is the choice \zeta_+
    xi   = - s - 1.j * sigma_p # This is the choice \xi_-
    eta  = -1.j * sigma_m      # This is the choice \eta_+

    return zeta, xi, eta

def D_coeffs(omega, a, s, m, A):
    """ TODO """

    zeta, xi, eta = sing_pt_char_exps(omega, a, s, m)

    root  = np.sqrt(1. - a*a)

    p     = root * zeta
    alpha = 1. + s + xi + eta - 2.*zeta + s # Because we took the root \zeta_+
    gamma = 1. + s + 2.*eta
    delta = 1. + s + 2.*xi
    sigma = (A + a*a*omega*omega - 8.*omega*omega
             + p * (2.*alpha + gamma - delta)
             + (1. + s - 0.5*(gamma + delta))
             * (s + 0.5*(gamma + delta)))

    D = [0] * 5
    D[0] = delta
    D[1] = 4.*p - 2.*alpha + gamma - delta - 2.
    D[2] = 2.*alpha - gamma + 2.
    D[3] = alpha*(4.*p - delta) - sigma
    D[4] = alpha*(alpha - gamma + 1.)

    return D

def leaver_cf_trunc_inversion(omega, a, s, m, A,
                              n_inv, N=300, r_N=1.):
    """ Approximate the n_inv inversion of the infinite continued
    fraction for solving the radial Teukolsky equation, using
    N terms total for the approximation. This uses "bottom up"
    evaluation, and you can pass a seed value r_N to assume for
    the rest of the infinite fraction which has been truncated.
    The value returned is Eq. (44).
    TODO seriously document this! """

    n = np.arange(0, N+1)

    D = D_coeffs(omega, a, s, m, A)

    alpha =     n*n + (D[0] + 1.)*n + D[0]
    beta  = -2.*n*n + (D[1] + 2.)*n + D[3]
    gamma =     n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.

    conv1 = 0.
    for i in range(0, n_inv): # n_inv is not included
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    conv2 = -r_N # Is this sign correct?
    for i in range(N, n_inv, -1): # n_inv is not included
        conv2 = gamma[i] / (beta[i] - alpha[i] * conv2)

    return (beta[n_inv]
            - gamma[n_inv] * conv1
            - alpha[n_inv] * conv2)

# TODO possible choices for r_N: 0., 1., approximation using (34)-(38)

def leaver_cf_inv_lentz(omega, a, s, m, A, n_inv,
                        tol=1.e-10, N_min=0, N_max=np.Inf):
    """ Compute the n_inv inversion of the infinite continued
    fraction for solving the radial Teukolsky equation, using
    modified Lentz's method.
    The value returned is Eq. (44).
    TODO seriously document this! """

    D = D_coeffs(omega, a, s, m, A)

    # This is only use for the terminating fraction
    n = np.arange(0, n_inv+1)
    alpha =     n*n + (D[0] + 1.)*n + D[0]
    beta  = -2.*n*n + (D[1] + 2.)*n + D[3]
    gamma =     n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.

    conv1 = 0.
    for i in range(0, n_inv): # n_inv is not included
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    # In defining the below a, b sequences, I have cleared a fraction
    # compared to the usual way of writing the radial infinite
    # continued fraction. The point of doing this was that so both
    # terms, a(n) and b(n), tend to 1 as n goes to infinity. Further,
    # We can analytically divide through by n in the numerator and
    # denominator to make the numbers closer to 1.
    def a(i):
        n = i + n_inv - 1
        return -(n*n + (D[0] + 1.)*n + D[0])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)

    def b(i):
        if (i==0): return 0
        n = i + n_inv
        return (-2.*n*n + (D[1] + 2.)*n + D[3])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)

    conv2, cf_err, n_frac = lentz(a, b, tol=tol, N_min=N_min, N_max=N_max)

    return (beta[n_inv]
            - gamma[n_inv] * conv1
            + gamma[n_inv] * conv2), cf_err, n_frac

# TODO possible choices for r_N: 0., 1., approximation using (34)-(38)
