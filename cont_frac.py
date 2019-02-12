from __future__ import division, print_function, absolute_import

import numpy as np

# TODO some documentation here, better documentation throughout

def Lentz(a, b, tol=1.e-10, N_min=0, N_max=np.Inf, tiny=1.e-30):
    """ Compute a continued fraction via modified Lentz's method.

    TODO Documentation
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
