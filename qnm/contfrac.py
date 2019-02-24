""" Infinite continued fractions via Lentz's method.

TODO Documentation.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

# TODO some documentation here, better documentation throughout

def lentz(a, b, tol=1.e-10, N_min=0, N_max=np.Inf, tiny=1.e-30):
    """ Compute a continued fraction via modified Lentz's method.

    This implementation is by the book [1]_.

    Parameters
    ----------
    a: callable returning numeric.
    b: callable returning numeric.

    tol: float [default: 1.e-10]
      Tolerance for termination of evaluation.

    N_min: int [default: 0]
      Minimum number of iterations to evaluate.

    N_max: int or comparable [default: np.Inf]
      Maximum number of iterations to evaluate.

    tiny: float [default: 1.e-30]
      Very small number to control convergence of Lentz's method when
      there is cancellation in a denominator.

    Returns
    -------
    (float, float, int)
      The first element of the tuple is the value of the continued
      fraction. The second element is the estimated error. The third
      element is the number of iterations.

    References
    ----------
    .. [1] WH Press, SA Teukolsky, WT Vetterling, BP Flannery,
       "Numerical Recipes," 3rd Ed., Cambridge University Press 2007,
       ISBN 0521880688, 9780521880688 .

    """

    f_old = b(0)

    if (f_old == 0):
        f_old = tiny

    C_old = f_old
    D_old = 0.

    conv = False

    j = 1

    while ((not conv) and (j < N_max)):

        aj, bj = a(j), b(j)

        D_new = bj + aj * D_old

        if (D_new == 0):
            D_new = tiny

        C_new = bj + aj / C_old

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

    # Success or failure can be assessed by the user
    return f_new, np.abs(Delta - 1.), j-1

def lentz_gen(a, b, tol=1.e-10, N_min=0, N_max=np.Inf, tiny=1.e-30):
    """ Compute a continued fraction via modified Lentz's method,
    using generators rather than functions.

    This implementation is by the book [1]_.

    Parameters
    ----------
    a: generator yielding numeric.
    b: generator yielding numeric.

    tol: float [default: 1.e-10]
      Tolerance for termination of evaluation.

    N_min: int [default: 0]
      Minimum number of iterations to evaluate.

    N_max: int or comparable [default: np.Inf]
      Maximum number of iterations to evaluate.

    tiny: float [default: 1.e-30]
      Very small number to control convergence of Lentz's method when
      there is cancellation in a denominator.

    Returns
    -------
    (float, float, int)
      The first element of the tuple is the value of the continued
      fraction. The second element is the estimated error. The third
      element is the number of iterations.

    References
    ----------
    .. [1] WH Press, SA Teukolsky, WT Vetterling, BP Flannery,
       "Numerical Recipes," 3rd Ed., Cambridge University Press 2007,
       ISBN 0521880688, 9780521880688 .

    """

    f_old = next(b) # 0

    if (f_old == 0):
        f_old = tiny

    C_old = f_old
    D_old = 0.

    conv = False

    j = 1

    while ((not conv) and (j < N_max)):

        aj, bj = next(a), next(b) # j and j. 'a' started with 1.

        D_new = bj + aj * D_old

        if (D_new == 0):
            D_new = tiny

        C_new = bj + aj / C_old

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

    # Success or failure can be assessed by the user
    return f_new, np.abs(Delta - 1.), j-1
