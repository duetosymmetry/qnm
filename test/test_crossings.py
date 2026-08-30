import numpy as np
import qnm
from qnm.crossings import find_min_gap, mode_gap


class TestCrossings(object):
    @classmethod
    def setup_class(cls):
        qnm.download_data()

    def test_known_225_226_crossing(self):
        """The (2,2,5)-(2,2,6) avoided crossing near a ~ 0.9."""
        a_star, gap = find_min_gap((-2, 2, 2, 5), (-2, 2, 2, 6),
                                   a_min=0.8, a_max=0.95)
        assert abs(a_star - 0.8975) < 5e-3
        assert abs(gap - 0.0667) < 5e-3

    def test_gap_array_and_scalar(self):
        spins = np.array([0.1, 0.5])
        gaps = mode_gap(spins, (-2, 2, 2, 0), (-2, 2, 2, 1))
        assert gaps.shape == (2,)
        assert np.all(gaps > 0)
        scalar = mode_gap(0.5, (-2, 2, 2, 0), (-2, 2, 2, 1))
        assert np.isclose(scalar, gaps[1])
