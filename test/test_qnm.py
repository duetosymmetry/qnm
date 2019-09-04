import pytest
import qnm
import numpy as np

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

    def test_example(self):
        """
        An example of a test
        """
        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        omega, A, C = grav_220(a=0.68)
        assert omega == (0.5239751042900845 - 0.08151262363119974j)

    def test_example2(self):
        from qnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz
        old = leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        new = leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        assert np.all([old[i] == new[i] for i in range(3)])
