from __future__ import division, print_function, absolute_import

import logging
import pickle
import os

import numpy as np

import angular
from Schw_n_seq_finder import Schw_n_seq_finder

# TODO some documentation here, better documentation throughout

def build_Schw_dict(*args, **kwargs):
    """ Function to build a dict of Schwarzschild QNMs.

    Loops over values of (s,l), using Schw_n_seq_finder to find
    sequences in n.

    TODO Documentation

    Keyword arguments
    =================
    s_arr: [int] [default: [-2, -1, 0]]
      Array of s values to run over.

    n_max: int [default: 20]
      Maximum overtone number to run over.

    l_max: int [default: 20]
      Maximum angular harmonic number to run over.

    tol: float [default: 1e-10]
      Tolerance to pass to Schw_n_seq_finder.

    Return
    ======
    TODO Document the format of the dict

    """

    # TODO: Should we allow any other params to be customizable?
    s_arr   = kwargs.get('s_arr',   [-2, -1, 0])
    n_max   = kwargs.get('n_max',   20)
    l_max   = kwargs.get('l_max',   20)

    tol     = kwargs.get('tol',     1e-10)

    Schw_dict = {}
    Schw_err_dict = {}

    for s in s_arr:
        ls = np.arange(angular.l_min(s,0),l_max)
        for l in ls:
            Schw_seq = Schw_n_seq_finder(s=s, l=l,
                                         l_max=l+1, # Angular matrix
                                         # will be diagonal for Schw
                                         n_max=n_max, tol=tol)
            try:
                Schw_seq.do_find_sequence()
            except:
                logging.warn("Failed at s={}, l={}".format(s, l))
            for n, (omega, cf_err, iters) in enumerate(zip(Schw_seq.omega,
                                                           Schw_seq.cf_err,
                                                           Schw_seq.iters)):
                Schw_dict[(s,l,n)] = (np.asscalar(omega),
                                      np.asscalar(cf_err),
                                      int(iters))
                Schw_dict[(-s,l,n)] = Schw_dict[(s,l,n)]

    return Schw_dict

############################################################

class Schw_QNM_dict(object):

    # Borg pattern, the QNM table will be shared among all instances
    _shared_state = {}

    def __init__(self, init=False):
        """ TODO Documentation! """

        self.__dict__ = self._shared_state

        if (not hasattr(self, 'Schw_QNM_dict')):
            # First!
            self.Schw_QNM_dict = None

        if (init):
            self.load_dict()


    def load_dict(self):
        """ Load a Schw QNM dict from disk, or compute one if needed.
        """
        if (self.Schw_QNM_dict is not None):
            return self.Schw_QNM_dict

        # TODO magic file name
        Schw_table_pickle_file = './data/Schw_table.pickle'

        try:
            with open(Schw_table_pickle_file, 'rb') as handle:
                logging.info("Loading Schw QNM dict from file {}".format(Schw_table_pickle_file))
                self.Schw_QNM_dict = pickle.load(handle)

        except:
            logging.info("Could not load Schw QNM dict from file, computing")
            # TODO no parameters allowed?
            self.Schw_QNM_dict = build_Schw_dict()

            _the_dir = os.path.dirname(Schw_table_pickle_file)
            if not os.path.exists(_the_dir):
                try:
                    os.mkdir(_the_dir)
                    with open(Schw_table_pickle_file, 'wb') as handle:
                        logging.info("Writing Schw QNM dict to file {}".format(Schw_table_pickle_file))
                        pickle.dump(self.Schw_QNM_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    logging.warn("Could not write Schw QNM dict to file {}".format(Schw_table_pickle_file))

        return self.Schw_QNM_dict
