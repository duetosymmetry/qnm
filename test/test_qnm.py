import pytest
import qnm
import numpy as np
from pathlib import Path

class TestQnm(object):
    """
    A class for testing aspects of the qnm module.
    """
    @classmethod
    def setup_class(cls):
        """
        Download the data when setting up the test class.
        """
        qnm.download_data()

    def test_cache_file_operations(self):
        """Test file operations and downloading the on-disk cache.
        """

        print("Downloading with overwrite=True")
        qnm.cached.download_data(overwrite=True)
        print("Clearing disk cache but not tarball")
        qnm.cached._clear_disk_cache(delete_tarball=False)
        print("Decompressing tarball")
        qnm.cached._decompress_data()

    def test_example(self):
        """
        An example of a test
        """
        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        omega, A, C = grav_220(a=0.68)
        assert omega == (0.5239751042900845 - 0.08151262363119974j)

    def test_example2(self):
        """ Check consistency between old and new Leaver solvers """
        from qnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz
        old = leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        new = leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        assert np.all([old[i] == new[i] for i in range(3)])

    def test_build_cache(self):
        """Check the default cache-building functionality"""

        qnm.cached._clear_disk_cache(delete_tarball=False)
        qnm.modes_cache.seq_dict = {}
        qnm.cached.build_package_default_cache(qnm.modes_cache)
        assert 861 == len(qnm.modes_cache.seq_dict.keys())
        qnm.modes_cache.write_all()
        assert 861 == len(list(qnm.cached.get_cachedir().glob('*.pickle')))
