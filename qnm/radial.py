""" Solve the radial Teukolsky equation via Leaver's method.

TODO Documentation.

.. Note that numba's decorators confuse autodoc. Therefore you must
   update docs/_autosummary/qnm.radial.rst if you add any functions
   that are decorated by numba.

"""

from __future__ import division, print_function, absolute_import

from numba import njit
import numpy as np

from .contfrac import lentz

# TODO some documentation here, better documentation throughout

@njit(cache=True)
def sing_pt_char_exps(omega, a, s, m):
    r""" Compute the three characteristic exponents of the singular points
    of the radial Teukolsky equation.

    We want ingoing at the outer horizon and outgoing at infinity. The
    choice of one of two possible characteristic exponents at the
    inner horizon doesn't affect the minimal solution in Leaver's
    method, so we just pick one. Thus our choices are, in the
    nomenclature of [1]_, :math:`(\zeta_+, \xi_-, \eta_+)`.

    Parameters
    ----------
    omega: complex
      The complex frequency in the ansatz for the solution of the
      radial Teukolsky equation.

    a: double
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    Returns
    -------
    (complex, complex, complex)
      :math:`(\zeta_+, \xi_-, \eta_+)`

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

    root = np.sqrt(1. - a*a)
    r_p, r_m = 1. + root, 1. - root

    sigma_p = (2.*omega*r_p - m*a)/(2.*root)
    sigma_m = (2.*omega*r_m - m*a)/(2.*root)

    zeta = +1.j * omega        # This is the choice \zeta_+
    xi   = - s - 1.j * sigma_p # This is the choice \xi_-
    eta  = -1.j * sigma_m      # This is the choice \eta_+

    return zeta, xi, eta

@njit(cache=True)
def D_coeffs(omega, a, s, m, A):
    """ The D_0 through D_4 coefficients that enter into the radial
    infinite continued fraction, Eqs. (31) of [1]_ .


    Parameters
    ----------
    omega: complex
      The complex frequency in the ansatz for the solution of the
      radial Teukolsky equation.

    a: double
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    A: complex
      Separation constant between angular and radial ODEs.

    Returns
    -------
    array[5] of complex
      D_0 through D_4 .

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

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

    D = [0.j] * 5
    D[0] = delta
    D[1] = 4.*p - 2.*alpha + gamma - delta - 2.
    D[2] = 2.*alpha - gamma + 2.
    D[3] = alpha*(4.*p - delta) - sigma
    D[4] = alpha*(alpha - gamma + 1.)

    return D

def leaver_cf_trunc_inversion(omega, a, s, m, A,
                              n_inv, N=300, r_N=1.):
    """ Legacy function.

    Approximate the n_inv inversion of the infinite continued
    fraction for solving the radial Teukolsky equation, using
    N terms total for the approximation. This uses "bottom up"
    evaluation, and you can pass a seed value r_N to assume for
    the rest of the infinite fraction which has been truncated.
    The value returned is Eq. (44) of [1]_.

    Parameters
    ----------
    omega: complex
      The complex frequency for evaluating the infinite continued
      fraction.

    a: float
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    A: complex
      Separation constant between angular and radial ODEs.

    n_inv: int
      Inversion number for the infinite continued fraction. Finding
      the nth overtone is typically most stable when n_inv = n .

    N: int, optional [default: 300]
      The depth where the infinite continued fraction is truncated.

    r_N: float, optional [default: 1.]
      Value to assume for the rest of the infinite continued fraction
      past the point of truncation.

    Returns
    -------
    complex
      The nth inversion of the infinite continued fraction evaluated
      with these arguments.

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

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

def leaver_cf_inv_lentz_old(omega, a, s, m, A, n_inv,
                            tol=1.e-10, N_min=0, N_max=np.Inf):
    """ Legacy function. Same as :meth:`leaver_cf_inv_lentz` except
    calling :meth:`qnm.contfrac.lentz` with temporary functions that
    are defined inside this function. Numba does not speed up
    this type of code. However it remains here for testing purposes.
    See documentation for :meth:`leaver_cf_inv_lentz` for parameters
    and return value.

    Examples
    --------

    >>> from qnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz
    >>> print(leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0))
    ((-3.5662773770495546-1.5388710793384461j), 9.702542314939062e-11, 76)

    Compare the two versions of the function:

    >>> old = leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
    >>> new = leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
    >>> [ old[i]-new[i] for i in range(3)]
    [0j, 0.0, 0]

    """

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


@njit(cache=True)
def leaver_cf_inv_lentz(omega, a, s, m, A, n_inv,
                               tol=1.e-10, N_min=0, N_max=np.Inf):
    """ Compute the n_inv inversion of the infinite continued
    fraction for solving the radial Teukolsky equation, using
    modified Lentz's method.
    The value returned is Eq. (44) of [1]_.

    Same as :meth:`leaver_cf_inv_lentz_old`, but with Lentz's method
    inlined so that numba can speed things up.

    Parameters
    ----------
    omega: complex
      The complex frequency for evaluating the infinite continued
      fraction.

    a: float
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    A: complex
      Separation constant between angular and radial ODEs.

    n_inv: int
      Inversion number for the infinite continued fraction. Finding
      the nth overtone is typically most stable when n_inv = n .

    tol: float, optional [default: 1.e-10]
      Tolerance for termination of Lentz's method.

    N_min: int, optional [default: 0]
      Minimum number of iterations through Lentz's method.

    N_max: int or comparable, optional [default: np.Inf]
      Maximum number of iterations for Lentz's method.

    Returns
    -------
    (complex, float, int)
      The first value (complex) is the nth inversion of the infinite
      continued fraction evaluated with these arguments. The second
      value (float) is the estimated error from Lentz's method. The
      third value (int) is the number of iterations of Lentz's method.

    Examples
    --------

    >>> from qnm.radial import leaver_cf_inv_lentz
    >>> print(leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0))
    ((-3.5662773770495546-1.5388710793384461j), 9.702542314939062e-11, 76)

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

    D = D_coeffs(omega, a, s, m, A)

    # This is only use for the terminating fraction
    n = np.arange(0, n_inv+1)
    alpha =     n*n + (D[0] + 1.)*n + D[0]
    beta  = -2.*n*n + (D[1] + 2.)*n + D[3]
    gamma =     n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.

    conv1 = 0.
    for i in range(0, n_inv): # n_inv is not included
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    ##############################
    # Beginning of Lentz's method, inlined

    # TODO should tiny be a parameter?
    tiny = 1.e-30

    # This is starting with b_0 = 0 for the infinite continued
    # fraction. I could have started with other values (e.g. b_i
    # evaluated with i=0) but then I would have had to subtract that
    # same quantity away from the final result. I don't know if this
    # affects convergence.
    f_old = tiny

    C_old = f_old
    D_old = 0.

    conv = False

    j = 1
    n = n_inv

    while ((not conv) and (j < N_max)):

        # In defining the below a, b sequences, I have cleared a fraction
        # compared to the usual way of writing the radial infinite
        # continued fraction. The point of doing this was that so both
        # terms, a(n) and b(n), tend to 1 as n goes to infinity. Further,
        # We can analytically divide through by n in the numerator and
        # denominator to make the numbers closer to 1.
        an = -(n*n + (D[0] + 1.)*n + D[0])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)
        n = n + 1
        bn = (-2.*n*n + (D[1] + 2.)*n + D[3])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)

        D_new = bn + an * D_old

        if (D_new == 0):
            D_new = tiny

        C_new = bn + an / C_old

        if (C_new == 0):
            C_new = tiny

        D_new = 1./D_new
        Delta = C_new * D_new
        f_new = f_old * Delta

        if ((j > N_min) and (np.abs(Delta - 1.) < tol)): # converged
            conv = True

        # Set up for next iter
        j = j + 1
        D_old = D_new
        C_old = C_new
        f_old = f_new

    conv2 = f_new

    ##############################
    
    return (beta[n_inv]
            - gamma[n_inv] * conv1
            + gamma[n_inv] * conv2), np.abs(Delta-1.), j-1

