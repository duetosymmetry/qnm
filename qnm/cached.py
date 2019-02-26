""" Caching interface to Kerr QNMs

TODO Documentation.
"""

from __future__ import division, print_function, absolute_import

import logging
import pickle
import os

import numpy as np

from .angular import l_min
from .spinsequence import KerrSpinSeq
from .schwarzschild.tabulated import QNMDict


def mode_pickle_path(s, l, m, n):
    """Construct the path to a pickle file for the mode (s, l, m, n)

    Parameters
    ----------
    s: int
      Spin-weight of field of interest.

    l: int
      Multipole number of mode. l >= angular.l_min(s, m)

    m: int
      Azimuthal number of mode.

    n: int
      Overtone number of mode.

    Returns
    -------
    string
      `<dirname of this file>/data/s<s>_l<l>_m<m>_n<n>.pickle`

     """

    assert l >= l_min(s, m), ("l={} must be >= l_min={}"
                              .format(l, l_min(s, m)))


    s_sign = '-' if (s<0) else ''
    m_sign = '-' if (m<0) else ''

    return os.path.abspath(
        '{}/data/s{}{}_l{}_m{}{}_n{}.pickle'.format(
            os.path.dirname(os.path.realpath(__file__)),
            s_sign, np.abs(s), l,
            m_sign, np.abs(m), n
        ))

def write_mode(spin_seq, pickle_path=None):
    """Write an instance of KerrSpinSeq to disk.

    Parameters
    ----------
    spin_seq: KerrSpinSeq
      The mode to write to disk.

    pickle_path: string, optional [default: None]
      Path to file to write. If None, get the path from
      :meth:`mode_pickle_path()`.

    Raises
    ------
    TODO

    """

    if (pickle_path is None):
        pickle_path = mode_pickle_path(spin_seq.s, spin_seq.l,
                                       spin_seq.m, spin_seq.n)

    _the_dir = os.path.dirname(pickle_path)
    if not os.path.exists(_the_dir):
        try:
            os.makedirs(_the_dir)
        except:
            logging.error("Could not create dir {} to store Kerr QNM sequence".format(_the_dir))

    try:
        with open(pickle_path, 'wb') as handle:
            logging.info("Writing Kerr QNM sequence to file {}".format(pickle_path))
            pickle.dump(spin_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        logging.error("Could not write Kerr QNM sequence to file {}".format(pickle_path))

    return

def load_cached_mode(s, l, m, n):
    """Read a KerrSpinSeq from disk.

    Path is determined by :meth:`mode_pickle_path(s, l, m, n)`.

    Parameters
    ----------
    s: int
      Spin-weight of field of interest.

    l: int
      Multipole number of mode. l >= angular.l_min(s, m)

    m: int
      Azimuthal number of mode.

    n: int
      Overtone number of mode.

    Returns
    -------
    KerrSpinSeq
      The mode, if it exists. Otherwise None.

    """

    spin_seq = None

    pickle_path = mode_pickle_path(s, l, m, n)

    try:
        with open(pickle_path, 'rb') as handle:
            logging.info("Loading Kerr QNM sequence from file {}".format(pickle_path))
            spin_seq = pickle.load(handle)
    except UnicodeDecodeError as e:
        with open(pickle_path, 'rb') as handle:
            logging.info("Loading Kerr QNM sequence from file {}".format(pickle_path))
            spin_seq = pickle.load(handle, encoding='latin1')
    except:
        logging.error("Could not load Kerr QNM sequence from file {}".format(pickle_path))

    return spin_seq

############################################################

class KerrSeqCache(object):
    """ TODO

    Attributes
    ----------
    TODO

    Parameters
    ----------
    init_schw: bool, optional [default: False]
      TODO

    compute_if_not_found: bool, optional [default: True]
      TODO

    """

    # Borg pattern, the QNM dict will be shared among all instances
    _shared_state = {}

    def __init__(self, init_schw=False, compute_if_not_found=True):

        self.__dict__ = self._shared_state

        if (not hasattr(self, 'schw_dict')):
            # First!
            self.schw_dict = QNMDict(init=init_schw)

        if (not hasattr(self, 'seq_dict')):
            # First!
            self.seq_dict = {}


    def __call__(self, s, l, m, n):
        """TODO

        """
