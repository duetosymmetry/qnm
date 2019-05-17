""" Caching interface to Kerr QNMs

This is a high-level interface to the package.  An instance of
:class:`KerrSeqCache` will return instances of
:class:`qnm.spinsequence.KerrSpinSeq` from memory or disk. If a spin
sequence is neither in memory nor on disk then it will be computed and
returned.

Use :meth:`download_data` to fetch a collection of precomputed spin
sequences from the web.

TODO More documentation.

"""

from __future__ import division, print_function, absolute_import

import logging
import pickle
import os
try:
    from urllib.request import urlretrieve # py 3
except ImportError:
    from urllib import  urlretrieve # py 2
from tqdm import tqdm
import tarfile
import glob

import numpy as np

from .angular import l_min
from .spinsequence import KerrSpinSeq
from .schwarzschild.tabulated import QNMDict

# TODO should all the functions be static member functions? No, I don't
# think so

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
        logging.error("Could not load Kerr QNM sequence from file {}. "
                      "Do you need to run qnm.download_data()?".format(pickle_path))

    return spin_seq

############################################################

class KerrSeqCache(object):
    """High-level caching interface for getting precomputed spin sequences.

    An instance of :class:`KerrSeqCache` will return instances of
    :class:`qnm.spinsequence.KerrSpinSeq` from memory or disk. If a
    spin sequence is neither in memory nor on disk then it will be
    computed and returned.

    Use :meth:`download_data` to fetch a collection of precomputed
    spin sequences from the web.

    Parameters
    ----------
    init_schw: bool, optional [default: False]
      Value of init flag to pass to
      :class:`qnm.schwarzschild.tabulated.QNMDict`. You should set this to
      True the first time in a session that you create a QNMDict
      (most likely via this class).

    compute_if_not_found: bool, optional [default: True]
      If a mode sequence is not found on disk, this flag controls
      whether to try to compute the sequence from scratch.

    compute_pars: dict, optional [default: None]
      A dict of parameters to pass to
      :class:`qnm.spinsequence.KerrSpinSeq` if computing a mode
      sequence from scratch.

    Examples
    --------

    >>> import qnm
    >>> # qnm.download_data() # Only need to do this once
    >>> ksc = qnm.cached.KerrSeqCache(init_schw=True) # Only need init_schw once per session
    >>> mode_seq = ksc(s=-2,l=2,m=2,n=0)
    >>> omega, A, C = mode_seq(a=0.68)
    >>> print(omega)
    (0.5239751042900845-0.08151262363119974j)

    """

    # Borg pattern, the QNM dict will be shared among all instances
    _shared_state = {}

    def __init__(self, init_schw=False, compute_if_not_found=True,
                 compute_pars=None):

        self.__dict__ = self._shared_state

        # Reset these attributes
        # TODO Is this the right thing to do? Maybe not?
        self.compute_if_not_found = compute_if_not_found
        self.compute_pars = dict(compute_pars) if compute_pars is not None else {}

        # Don't reset this one
        if (not hasattr(self, 'schw_dict')):
            # First!
            # The only reason for keeping this around is to know
            # whether or not QNMDict has been initialized
            # TODO FIGURE OUT IF WE SHOULD DO THIS
            self.schw_dict = QNMDict(init=init_schw)

        # Definitely don't reset this one
        if (not hasattr(self, 'seq_dict')):
            # First!
            self.seq_dict = {}


    def __call__(self, s, l, m, n,
                 compute_if_not_found=None,
                 compute_pars=None):
        """Load a :class:`qnm.spinsequence.KerrSpinSeq` from the cache
        or from disk if available.

        If the mode sequence is not available on disk, optionally
        compute it from scratch.
        TODO maybe take param: dict of params to pass to KerrSpinSeq?

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

        compute_if_not_found: bool, optional [default: self.compute_if_not_found]
          Whether or not to compute from scratch the spin sequence if
          it is not available on disk.

        compute_pars: dict, optional [default: self.compute_pars]
          Dict of parameters to pass to KerrSpinSeq if a mode sequence
          needs to be computed from scratch.

        Returns
        -------
        KerrSpinSeq
          The mode, if it is in the cache, on disk, or has been
          computed from scratch.  If the mode is not available and
          compute_if_not_found is false, return None.

        """

        assert l >= l_min(s, m), ("l={} must be >= l_min={}"
                                  .format(l, l_min(s, m)))

        assert n >= 0, ("n={} must be non-negative".format(n))

        if compute_if_not_found is None:
            compute_if_not_found = self.compute_if_not_found

        if compute_pars is None:
            compute_pars = self.compute_pars

        key = (s,l,m,n)
        if (key in self.seq_dict.keys()):
            return self.seq_dict[key]

        loaded_mode = load_cached_mode(s,l,m,n)

        if (loaded_mode is not None):
            self.seq_dict[key] = loaded_mode
            return loaded_mode

        if (compute_if_not_found):
            logging.info("mode not in cache, or on disk, trying to "
                         "compute s={}, l={}, m={}, n={}"
                         .format(s, l, m, n))
            the_pars = dict(compute_pars) # Make a copy
            the_pars.update({ 's': s, 'l': l, 'm': m, 'n': n })
            computed = KerrSpinSeq(**the_pars)
            computed.do_find_sequence()
            self.seq_dict[key] = computed
            return computed

        return None

    def write_all(self):
        """Write all of the modes in the cache to disk.

        TODO: Take an overwrite argument which will force overwrite or
        not.

        """

        for _, seq in self.seq_dict.items():
            write_mode(seq)

############################################################
def build_package_default_cache(ksc):
    """Compute the standard list of modes that this package
    promises to have in its cache.

    This method is intended to be used for building the modes from
    scratch in a predictable way.  If modes are available on disk then
    there will be no computation, simply loading all the default modes.

    Parameters
    ----------
    ksc: KerrSeqCache
      The cache that will hold the modes we are about to compute.

    Returns
    -------
    KerrSeqCache
      The updated cache.
    """

    QNMDict(init=True)

    a_max   = .9995
    tol     = 1e-10
    delta_a = 2.5e-3
    Nr_max  = 6000

    ksc.compute_pars.update({'a_max': a_max, 'tol': tol,
                             'delta_a': delta_a, 'Nr_max': Nr_max})

    reruns = []

    ns=np.arange(0,7)
    ss = [-2, -1]
    for s in ss:
        ls=np.arange(np.abs(s),8)
        for l in ls:
            ms=np.arange(-l,l+1)
            for m in ms:
                for n in ns:
                    try:
                        ksc(s, l, m, n)
                    except:
                        reruns.append((s,l,m,n))

    # Tweak the params a little bit for the reruns
    re2runs = []
    ksc.compute_pars.update({'delta_a': 1.9e-3})
    for s, l, m, n in reruns:
        try:
            ksc(s, l, m, n)
        except:
            re2runs.append((s, l, m, n))

    # Experimentally determined we only need one more case of reruns
    ksc.compute_pars.update({'a_max': .9992, 'delta_a': 1.8e-3})

    for s, l, m, n in re2runs:
        ksc(s, l, m, n)

    return ksc

############################################################
# This is taken verbatim from the tqdm examples, see
# https://pypi.org/project/tqdm/#hooks-and-callbacks

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

############################################################
def download_data(overwrite=False):
    """Fetch and decompress tarball of precomputed spin sequence from
    the web.

    Parameters
    ----------
    overwrite: bool, optional [default: False]
      If there is already a tarball on disk, this flag controls
      whether or not it is overwritten.

    """

    data_url = 'https://duetosymmetry.com/files/qnm/data.tar.bz2'
    filename = data_url.split('/')[-1]
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dest     = base_dir + '/' + filename

    if (os.path.exists(dest) and (overwrite is False)):
        print("Destination path {} already exists, use overwrite=True "
              "to force an overwrite.".format(dest))
        return

    print("Trying to fetch {}".format(data_url))
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                  desc=filename) as t:
        urlretrieve(data_url, filename=dest, reporthook=t.update_to)

    print("Trying to decompress file {}".format(dest))
    with tarfile.open(dest, "r:bz2") as tar:
        tar.extractall(base_dir)

    data_dir = base_dir + '/data'
    pickle_files = glob.glob(data_dir + '/*.pickle')
    print("Data directory {} contains {} pickle files"
          .format(data_dir, len(pickle_files)))
