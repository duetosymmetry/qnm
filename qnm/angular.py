""" Solve the angular Teukolsky equation via spectral decomposition.

For a given complex QNM frequency ω, the separation constant and
spherical-spheroidal decomposition are found as an eigenvalue and
eigenvector of an (infinite) matrix problem.  The interface to solving
this problem is :meth:`C_and_sep_const_closest`, which returns a
certain eigenvalue A and eigenvector C.  The eigenvector contains the
C coefficients in the equation

.. math:: {}_s Y_{\ell m}(\\theta, \phi; a\omega) = {\sum_{\ell'=\ell_{\min} (s,m)}^{\ell_\max}} C_{\ell' \ell m}(a\omega)\ {}_s Y_{\ell' m}(\\theta, \phi) \,.

Here ℓmin=max(\|m\|,\|s\|) (see :meth:`l_min`), and ℓmax can be chosen at
run time. The C coefficients are returned as a complex ndarray, with
the zeroth element corresponding to ℓmin.  You can get the associated
ℓ values by calling :meth:`ells`.

TODO More documentation.
"""

from __future__ import division, print_function, absolute_import

from numba import njit
import numpy as np

# TODO some documentation here, better documentation throughout

@njit(cache=True)
def _calF(s, l, m):
    """ Eq. (52b) """

    if ((0==s) and (0 == l+1)):
        # This can only happen when solving for the mode labeled by s=0, l=0, m=0
        return 0.

    return (np.sqrt( ((l+1)**2 - m*m) / (2*l+3) / (2*l+1) )
            * np.sqrt( ( (l+1)**2  - s*s)  / (l+1)**2 ))

@njit(cache=True)
def _calG(s, l, m):
    """ Eq. (52c) """
    if (0 == l):
        return 0.

    return np.sqrt( ( l*l - m*m ) / (4*l*l - 1)) * np.sqrt(1 - s*s/l/l)

@njit(cache=True)
def _calH(s, l, m):
    """ Eq. (52d) """
    if (0 == l) or (0 == s):
        return 0.

    return - m*s/l/(l+1)

@njit(cache=True)
def _calA(s, l, m):
    """ Eq. (53a) """
    return _calF(s,l,m) * _calF(s,l+1,m)

@njit(cache=True)
def _calD(s, l, m):
    """ Eq. (53b) """
    return _calF(s,l,m) * (_calH(s,l+1,m)  + _calH(s,l,m))

@njit(cache=True)
def _calB(s, l, m):
    """ Eq. (53c) """
    return (_calF(s,l,m) * _calG(s,l+1,m)
            + _calG(s,l,m) * _calF(s,l-1,m)
            + _calH(s,l,m)**2)

@njit(cache=True)
def _calE(s, l, m):
    """ Eq. (53d) """
    return _calG(s,l,m) * (_calH(s,l-1,m) + _calH(s,l,m))

@njit(cache=True)
def _calC(s, l, m):
    """ Eq. (53e) """
    return _calG(s,l,m) * _calG(s,l-1,m)

@njit(cache=True)
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

@njit(cache=True)
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
        return -c*c*_calA(s,lprime,m)
    if (lprime == l-1):
        return (-c*c*_calD(s,lprime,m)
                + 2*c*s*_calF(s,lprime,m))
    if (lprime == l  ):
        return (swsphericalh_A(s,lprime,m)
                - c*c*_calB(s,lprime,m)
                + 2*c*s*_calH(s,lprime,m))
    if (lprime == l+1):
        return (-c*c*_calE(s,lprime,m)
                + 2*c*s*_calG(s,lprime,m))
    if (lprime == l+2):
        return -c*c*_calC(s,lprime,m)

    return 0.

def give_M_matrix_elem_ufunc(s, c, m):
    """Gives ufunc that implements matrix elements of the
    spherical-spheroidal decomposition matrix. This function is used
    by :meth:`M_matrix_old` and can be considered legacy.

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

@njit(cache=True)
def l_min(s, m):
    """ Minimum allowed value of l for a given s, m.

    The formula is l_min = max(\|m\|,\|s\|).

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

    return max(abs(s), abs(m))

@njit(cache=True)
def ells(s, m, l_max):
    """Vector of ℓ values in C vector and M matrix.

    The format of the C vector and M matrix is that the 0th element
    corresponds to l_min(s,m) (see :meth:`l_min`).

    Parameters
    ----------
    s: int
      Spin-weight of interest

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    int ndarray
      Vector of ℓ values, starting from l_min
    """

    return np.arange(l_min(s,m), l_max+1)

def M_matrix_old(s, c, m, l_max):
    """Legacy function. Same as :meth:`M_matrix` except trying to be cute
    with ufunc's, requiring scope capture with temp func inside
    :meth:`give_M_matrix_elem_ufunc`, which meant that numba could not
    speed up this method. Remains here for testing purposes. See
    documentation for :meth:`M_matrix` parameters and return value.
    """

    _ells = ells(s, m, l_max)

    uf = give_M_matrix_elem_ufunc(s, c, m)

    return uf.outer(_ells,_ells).astype(complex)

@njit(cache=True)
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

    _ells = ells(s, m, l_max)

    M = np.empty((len(_ells),len(_ells)), dtype=np.complex128)

    for i in range(len(_ells)):
        for j in range(len(_ells)):
            M[i,j] = M_matrix_elem(s, c, m, _ells[i], _ells[j])

    return M

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
      corresponding eigenvector.  The 0th element of this ndarray
      corresponds to :meth:`l_min`.
    """

    As, Cs = np.linalg.eig(M_matrix(s, c, m, l_max))
    i_closest = np.argmin(np.abs(As-A0))
    return As[i_closest], Cs[:,i_closest]
