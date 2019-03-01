""" Solve the angular Teukolsky equation via spectral decomposition.

TODO Documentation.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

# TODO some documentation here, better documentation throughout

def calF(s, l, m):
    """ Eq. (52b) """

    if ((0==s) and (0 == l+1)):
        # This can only happen when solving for the mode labeled by s=0, l=0, m=0
        return 0.

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

def swsphericalh_A(s, l, m):
    """ Angular separation constant at a=0.

    Eq. (50). Has no dependence on m. The formula is
      A_0 = l(l+1) - s(s+1)

    Parameters
    ----------
    s: int
      Spin-weight of interest

    l: int
      Angular quantum number of interest

    m: int
      Magnetic quantum number, ignored

    Returns
    -------
    int
      Value of A(a=0) = l(l+1) - s(s+1)
    """

    return l*(l+1) - s*(s+1)

def M_matrix_elem(s, c, m, l, lprime):
    """ The (l, lprime) matrix element from the spherical-spheroidal
    decomposition matrix from Eq. (55).

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of the spheroidal harmonic

    m: int
      Magnetic quantum number

    l: int
      Angular quantum number of interest

    lprime: int
      Primed quantum number of interest

    Returns
    -------
    complex
      Matrix element M_{l, lprime}
    """

    if (lprime == l-2):
        return -c*c*calA(s,lprime,m)
    if (lprime == l-1):
        return (-c*c*calD(s,lprime,m)
                + 2*c*s*calF(s,lprime,m))
    if (lprime == l  ):
        return (swsphericalh_A(s,lprime,m)
                - c*c*calB(s,lprime,m)
                + 2*c*s*calH(s,lprime,m))
    if (lprime == l+1):
        return (-c*c*calE(s,lprime,m)
                + 2*c*s*calG(s,lprime,m))
    if (lprime == l+2):
        return -c*c*calC(s,lprime,m)

    return 0.

# I don't know if this is necessary ... can just iterate
def give_M_matrix_elem_ufunc(s, c, m):
    """Gives ufunc that implements matrix elements of the
    spherical-spheroidal decomposition matrix.

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of the spheroidal harmonic

    m: int
      Magnetic quantum number

    Returns
    -------
    ufunc
      Implements elements of M matrix
    """

    def elem(l, lprime):
        return M_matrix_elem(s, c, m, l, lprime)

    return np.frompyfunc(elem, 2, 1)

def l_min(s, m):
    """ Minimum allowed value of l for a given s, m.

    The formula is l_min = max(|m|,|s|).

    Parameters
    ----------
    s: int
      Spin-weight of interest

    m: int
      Magnetic quantum number

    Returns
    -------
    int
      l_min
    """

    return np.max([np.abs(s), np.abs(m)])

def M_matrix(s, c, m, l_max):
    """Spherical-spheroidal decomposition matrix truncated at l_max.

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of the spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex ndarray
      Decomposition matrix
    """

    ells = np.arange(l_min(s,m), l_max+1)

    uf = give_M_matrix_elem_ufunc(s, c, m)

    return uf.outer(ells,ells).astype(complex)

def sep_consts(s, c, m, l_max):
    """Finds eigenvalues of decomposition matrix, i.e. the separation
    constants, As.

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex ndarray
      Eigenvalues of spherical-spheroidal decomposition matrix
    """

    return np.linalg.eigvals(M_matrix(s, c, m, l_max))

def sep_const_closest(A0, s, c, m, l_max):
    """Gives the separation constant that is closest to A0.

    Parameters
    ----------
    A0: complex
      Value close to the desired separation constant.

    s: int
      Spin-weight of interest

    c: complex
      Oblateness of spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex
      Separation constant that is the closest to A0.
    """

    As = sep_consts(s, c, m, l_max)
    i_closest = np.argmin(np.abs(As-A0))
    return As[i_closest]

def C_and_sep_const_closest(A0, s, c, m, l_max):
    """Get a single eigenvalue and eigenvector of decomposition
    matrix, where the eigenvalue is closest to some guess A0.

    Parameters
    ----------
    A0: complex
      Value close to the desired separation constant.

    s: int
      Spin-weight of interest

    c: complex
      Oblateness of spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex, complex ndarray
      The first element of the tuple is the eigenvalue that is closest
      in value to A0. The second element of the tuple is the
      corresponding eigenvector.
    """

    As, Cs = np.linalg.eig(M_matrix(s, c, m, l_max))
    i_closest = np.argmin(np.abs(As-A0))
    return As[i_closest], Cs[:,i_closest]
