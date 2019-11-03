import pytest
import qnm
import numpy as np
try:
    from pathlib import Path # py 3
except ImportError:
    from pathlib2 import Path # py 2

class QnmTestDownload(object):
    """
    Base class so that each test will automatically download_data
    """
    @classmethod
    def setup_class(cls):
        """
        Download the data when setting up the test class.
        """
        qnm.download_data()

class TestQnmFileOps(QnmTestDownload):
    def test_cache_file_operations(self):
        """Test file operations and downloading the on-disk cache.
        """

        print("Downloading with overwrite=True")
        qnm.cached.download_data(overwrite=True)
        print("Clearing disk cache but not tarball")
        qnm.cached._clear_disk_cache(delete_tarball=False)
        print("Decompressing tarball")
        qnm.cached._decompress_data()

class TestQnmOneMode(QnmTestDownload):
    def test_one_mode(self):
        """
        An example of a test
        """
        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        omega, A, C = grav_220(a=0.68)
        assert np.allclose(omega, (0.5239751042900845 - 0.08151262363119974j))

class TestQnmNewLeaverSolver(QnmTestDownload):
    def test_compare_old_new_Leaver(self):
        """ Check consistency between old and new Leaver solvers """
        from qnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz
        old = leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        new = leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        assert np.all([old[i] == new[i] for i in range(3)])

class TestQnmSolveInterface(QnmTestDownload):
    """
    Test the various interface options for solving
    """

    def test_interp_only(self):
        """Check that we get reasonable values (but not identical!)
        with just interpolation.
        """

        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        a = 0.68
        assert a not in grav_220.a

        omega_int, A_int, C_int = grav_220(a=a, interp_only=True)
        omega_sol, A_sol, C_sol = grav_220(a=a, interp_only=False, store=False)

        assert np.allclose(omega_int, omega_sol) and not np.equal(omega_int, omega_sol)
        assert np.allclose(A_int, A_sol) and not np.equal(A_int, A_sol)
        assert np.allclose(C_int, C_sol) and not all(np.equal(C_int, C_sol))

    def test_store_a(self):
        """Check that the option store=True updates a spin sequence"""

        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)

        old_n = len(grav_220.a)
        k = int(old_n/2)

        new_a = 0.5 * (grav_220.a[k] + grav_220.a[k+1])

        assert new_a not in grav_220.a

        _, _, _ = grav_220(new_a, store=False)
        n_1 = len(grav_220.a)
        assert old_n == n_1

        _, _, _ = grav_220(new_a, store=True)
        n_2 = len(grav_220.a)
        assert n_2 == n_1 + 1

    def test_resolve(self):
        """Test that option resolve_if_found=True really does a new
        solve"""

        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)

        n = len(grav_220.a)
        k = int(n/2)
        a = grav_220.a[k]

        omega_old, A_old, C_old = grav_220(a=a, resolve_if_found=False)

        old_tol = grav_220.solver.tol
        old_cf_tol = grav_220.solver.cf_tol

        grav_220.solver.set_params(tol = old_tol * 0.1, cf_tol = old_cf_tol * 0.1)

        omega_new, A_new, C_new = grav_220(a=a, resolve_if_found=True)

        assert np.allclose(omega_new, omega_old) and not np.equal(omega_new, omega_old)
        assert np.allclose(A_new, A_old) and not np.equal(A_new, A_old)
        assert np.allclose(C_new, C_old) and not all(np.equal(C_new, C_old))

class TestQnmBuildCache(QnmTestDownload):
    def test_build_cache(self):
        """Check the default cache-building functionality"""

        qnm.cached._clear_disk_cache(delete_tarball=False)
        qnm.modes_cache.seq_dict = {}
        qnm.cached.build_package_default_cache(qnm.modes_cache)
        assert 860 == len(qnm.modes_cache.seq_dict.keys())
        qnm.modes_cache.write_all()
        cache_data_dir = qnm.cached.get_cachedir() / 'data'

        # Magic number, default num modes is 860
        assert 860 == len(list(cache_data_dir.glob('*.pickle')))
