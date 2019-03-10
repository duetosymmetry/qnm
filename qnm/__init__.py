"""Calculate quasinormal modes of Kerr black holes.

The highest-level interface is via :class:`qnm.cached.KerrSeqCache`,
which will fetch instances of
:class:`qnm.spinsequence.KerrSpinSeq`. This is most clearly
demonstrated with an example.

TODO More documentation

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

from __future__ import print_function, division, absolute_import

from ._version import __version__

__copyright__ = "Copyright (C) 2019 Leo C. Stein"
__email__ = "leo.stein@gmail.com"
__status__ = "testing"
__author__ = "Leo C. Stein"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from . import radial
from . import angular
from . import contfrac
from . import nearby

from . import spinsequence

from . import cached
from .cached import download_data
