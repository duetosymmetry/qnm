""" Computing, loading, and storing tabulated Schwarzschild QNMs.

TODO Documentation.
"""

from __future__ import division, print_function, absolute_import

import logging
import pickle
import os

import numpy as np

from ..angular import l_min
from .overtonesequence import SchwOvertoneSeq

# TODO some documentation here, better documentation throughout

def build_Schw_dict(*args, **kwargs):
    """ Function to build a dict of Schwarzschild QNMs.

    Loops over values of (s,l), using SchwOvertoneSeq to find
    sequences in n.

    TODO Documentation

    Parameters
    ----------
    s_arr: [int] [default: [-2, -1, 0]]
      Array of s values to run over.

    n_max: int [default: 20]
      Maximum overtone number to run over.

    l_max: int [default: 20]
      Maximum angular harmonic number to run over.

    tol: float [default: 1e-10]
      Tolerance to pass to SchwOvertoneSeq.

    Returns
    -------
    dict
      A dict with tuple keys (s,l,n).
      The value at d[s,l,n] is a tuple (omega, cf_err, n_frac)
      where omega is the frequency omega_{s,l,n}, cf_err is the
      estimated truncation error for the continued fraction, and
      n_frac is the depth of the continued fraction evaluation.

    """

    # TODO: Should we allow any other params to be customizable?
    s_arr   = kwargs.get('s_arr',   [-2, -1, 0])
    n_max   = kwargs.get('n_max',   20)
    l_max   = kwargs.get('l_max',   20)

    tol     = kwargs.get('tol',     1e-10)

    Schw_dict = {}
    Schw_err_dict = {}

    for s in s_arr:
        ls = np.arange(l_min(s,0),l_max)
        for l in ls:
            Schw_seq = SchwOvertoneSeq(s=s, l=l,
                                       l_max=l+1, # Angular matrix will be diagonal for Schw
                                       n_max=n_max, tol=tol)
            try:
                Schw_seq.do_find_sequence()
            except:
                logging.warn("Failed at s={}, l={}".format(s, l))
            for n, (omega, cf_err, n_frac) in enumerate(zip(Schw_seq.omega,
                                                            Schw_seq.cf_err,
                                                            Schw_seq.n_frac)):
                Schw_dict[(s,l,n)] = (np.asscalar(omega),
                                      np.asscalar(cf_err),
                                      int(n_frac))
                Schw_dict[(-s,l,n)] = Schw_dict[(s,l,n)]

    return Schw_dict

############################################################

class QNMDict(object):
    """ TODO Documentation! """

    # Borg pattern, the QNM table will be shared among all instances
    _shared_state = {}

    def __init__(self, init=False, dict_pickle_file=None):

        self.__dict__ = self._shared_state

        if (not hasattr(self, 'qnm_dict')):
            # First!
            self.qnm_dict = None

        if (init):
            self.load_dict(dict_pickle_file=dict_pickle_file)


    def load_dict(self, dict_pickle_file=None):
        """ Load a Schw QNM dict from disk, or compute one if needed.

        If a QNM dict has previously been loaded by any instance of
        the class, that dict will be returned immediately. If no QNM
        dict has been loaded, attempt to read it from disk. If that
        fails then a new table will be computed, written to the
        specified filename, and returned.

        Params
        ------
        dict_pickle_file: string [default: <dirname of this file>/data/Schw_dict.pickle]
          Filename for reading (or writing) dict of Schwarzschild QNMs

        Returns
        -------
        dict
          Same as :meth:`build_Schw_dict`.
          A dict with tuple keys (s,l,n).
          The value at d[s,l,n] is a tuple (omega, cf_err, n_frac)
          where omega is the frequency omega_{s,l,n}, cf_err is the
          estimated truncation error for the continued fraction, and
          n_frac is the depth of the continued fraction evaluation.
        """
        if (self.qnm_dict is not None):
            return self.qnm_dict

        if (dict_pickle_file is None):
            dict_pickle_file = os.path.abspath(
                '{}/data/Schw_dict.pickle'.format(
                    os.path.dirname(os.path.realpath(__file__))))

        try:
            with open(dict_pickle_file, 'rb') as handle:
                logging.info("Loading Schw QNM dict from file {}".format(dict_pickle_file))
                self.qnm_dict = pickle.load(handle)

        except:
            logging.info("Could not load Schw QNM dict from file {}, computing".format(dict_pickle_file))
            # TODO no parameters allowed?
            self.qnm_dict = build_Schw_dict()

            _the_dir = os.path.dirname(dict_pickle_file)
            if not os.path.exists(_the_dir):
                try:
                    os.mkdir(_the_dir)
                    with open(dict_pickle_file, 'wb') as handle:
                        logging.info("Writing Schw QNM dict to file {}".format(dict_pickle_file))
                        pickle.dump(self.qnm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    logging.warn("Could not write Schw QNM dict to file {}".format(dict_pickle_file))

        return self.qnm_dict
