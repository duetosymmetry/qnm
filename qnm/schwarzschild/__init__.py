""" Finding QNMs of Schwarzschild (numerically and approximately).

Schwarzschild QNMs are used as starting points for following a Kerr
QNM along a "spin sequence" in :class:`qnm.spinsequence.KerrSpinSeq`.
Most end users will not need to directly use the functions in this
module.
"""

from __future__ import print_function, division, absolute_import

from . import approx
from . import tabulated
from . import overtonesequence
