from __future__ import division, print_function, absolute_import

import numpy as np

# TODO some documentation here, better documentation throughout

def calF(s, l, m):
    """ Eq. (52b) """
    return (np.sqrt( ((l+1)**2 - m*m) / (2*l+3) / (2*l+1) )
            * np.sqrt( ( (l+1)**2  - s*s)  / (l+1)**2 ))

def calG(s, l, m):
    """ Eq. (52c) """
    if (0 == l):
        return 0.

    return np.sqrt( ( l*l - m*m ) / (4*l*l - 1)) * np.sqrt(1 - s*s/l/l)

def calH(s, l, m):
    """ Eq. (52d) """
    if (0 == l) or (0 == s):
        return 0.

    return - m*s/l/(l+1)

def calA(s, l, m):
    """ Eq. (53a) """
    return calF(s,l,m) * calF(s,l+1,m)

def calD(s, l, m):
    """ Eq. (53b) """
    return calF(s,l,m) * (calH(s,l+1,m)  + calH(s,l,m))

def calB(s, l, m):
    """ Eq. (53c) """
    return (calF(s,l,m) * calG(s,l+1,m)
            + calG(s,l,m) * calF(s,l-1,m)
            + calH(s,l,m)**2)

def calE(s, l, m):
    """ Eq. (53d) """
    return calG(s,l,m) * (calH(s,l-1,m) + calH(s,l,m))

def calC(s, l, m):
    """ Eq. (53e) """
    return calG(s,l,m) * calG(s,l-1,m)

def SWSphericalH_A(s, l, m):
    """ Eq. (50). Value at c=0. Has no dependence on m. """
    return l*(l+1) - s*(s+1)

def M_matrix_elem(s, c, m, l, lprime):
    """ Eq. (55) """

    # Notice that the M matrix is pentadiagonal,
    # so no need to write the sub- and super-diagonals
    # separately
    if (lprime == l-2) or (lprime == l+2):
        return -c*c*calA(s,lprime,m)
    if (lprime == l-1) or (lprime == l+1):
        return (-c*c*calD(s,lprime,m)
                + 2*c*s*calF(s,lprime,m))
    if (lprime == l  ):
        return (SWSphericalH_A(s,lprime,m)
                - c*c*calB(s,lprime,m)
                + 2*c*s*calH(s,lprime,m))

    return 0.

# I don't know if this is necessary ... can just iterate
def give_M_matrix_elem_ufunc(s, c, m):
    def elem(l, lprime):
        return M_matrix_elem(s, c, m, l, lprime)

    return np.frompyfunc(elem, 2, 1)

def M_matrix(s, c, m, l_max):
    l_min = np.max([np.abs(s), np.abs(m)])
    ells = np.arange(l_min, l_max+1)

    uf = give_M_matrix_elem_ufunc(s, c, m)

    return uf.outer(ells,ells).astype(complex)

def sep_consts(s, c, m, l_max):
    return np.linalg.eigvals(M_matrix(s, c, m, l_max))

def sep_const_closest(A0, s, c, m, l_max):
    As = sep_consts(s, c, m, l_max)
    i_closest = np.argmin(np.abs(As-A0))
    return As[i_closest]
