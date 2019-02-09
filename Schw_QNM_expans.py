from __future__ import division, print_function, absolute_import

import numpy as np

def Dolan_Ottewill_expansion(s, n, l):
    """ TODO documentation """

    L    = l + 0.5
    N    = n + 0.5
    beta = 1. - s*s

    DO_coeffs = { -1: 1.,
                   0: -1.j * N,
                  +1: beta/3. - 5.*N*N/36. - 115./432.,
                  +2: -1.j * N * (beta/9. + 235.*N*N/3888. - 1415./15552.),
                  +3: (-beta*beta/27. + (204.*N*N + 211.)/3888.*beta
                       +(854160.*N**4 - 1664760.*N*N - 776939.)/40310784.),
                  +4: -1.j * N *(beta*beta/27. + (1100.*N*N-2719.)/46656.*beta
                                 + (11273136.*N**4 - 52753800.*N*N + 66480535.)/2902376448.) }

    omega = 0.
    for k, c in DO_coeffs.items():
        omega = omega + c * np.power(L, -k)

    omega = omega / np.sqrt(27)

    return omega
