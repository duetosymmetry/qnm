[![github](https://img.shields.io/badge/GitHub-qnm-blue.svg)](https://github.com/duetosymmetry/qnm)
[![PyPI version](https://badge.fury.io/py/qnm.svg)](https://badge.fury.io/py/qnm)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/duetosymmetry/qnm/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/qnm/badge/?version=latest)](https://qnm.readthedocs.io/en/latest/?badge=latest)


# Welcome to qnm
Python implementation of Cook-Zalutskiy spectral approach to computing Kerr QNM frequencies.

TODO basic info

## Installation

### PyPI
_**qnm**_ is available through [PyPI](https://pypi.org/project/qnm/):

```shell
pip install qnm
```

### From source

```shell
git clone https://github.com/duetosymmetry/qnm.git
cd qnm
python setup.py install
```

If you do not have root permissions, replace the last step with
`python setup.py install --user`

## Dependencies
All of these can be installed through pip or conda.
* [numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [scipy](https://www.scipy.org/install.html)
* [numba](http://numba.pydata.org/numba-doc/latest/user/installing.html)

## Documentation

Automatically-generated API documentation is available on [Read the Docs: qnm](https://qnm.readthedocs.io/).


## Usage

The highest-level interface is via `qnm.cached.KerrSeqCache`, which
load cached *spin sequences* from disk. A spin sequence is just a mode
labeled by (s,l,m,n), with the spin a ranging from a=0 to some
maximum, e.g. 0.9995. A large number of low-lying spin sequences have
been precomputed and are available online. The first time you use the
package, download the precomputed sequences:

```python
>>> import qnm

>>> qnm.download_data() # Only need to do this once
Trying to fetch https://duetosymmetry.com/files/qnm/data.tar.bz2
Trying to decompress file /<something>/qnm/data.tar.bz2
Data directory /<something>/qnm/data contains 860 pickle files
```

Then, use `qnm.cached.KerrSeqCache` to load a
`qnm.spinsequence.KerrSpinSeq` of interest. If the mode is not
available, it will try to compute it (see detailed documentation for
how to control that calculation).

```python
>>> ksc = qnm.cached.KerrSeqCache(init_schw=True) # Only need init_schw once
>>> mode_seq = ksc(s=-2,l=2,m=2,n=0)
>>> omega, A, C = mode_seq(a=0.68)
>>> print(omega)
(0.5239751042900845-0.08151262363119974j)
```

Calling a spin sequence with `mode_seq(a)` will return the complex
quasinormal mode frequency omega, the complex angular separation
constant A, and a vector C of coefficients for decomposing the
associated spin-weighted spheroidal harmonics as a sum of
spin-weighted spherical harmonics.

## Credits
The code is developed and maintained by [Leo C. Stein](https://duetosymmetry.com).
