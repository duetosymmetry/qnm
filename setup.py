'''
Standard setup.py to upload the code on pypi.

    python setup.py sdist bdist_wheel
    twine upload dist/*
'''
import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode("UTF-8")

import sys
sys.path.append("qnm")

from _version import __version__

setuptools.setup(
    name="qnm",
    version=__version__,
    author="Leo C. Stein",
    author_email="leo.stein@gmail.com",
    description="Package for computing Kerr quasinormal mode frequencies, separation constants, and spherical-spheroidal mixing coefficients",
    keywords='black holes quasinormal modes physics scientific computing numerical methods',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duetosymmetry/qnm/",
    project_urls={
        "Bug Tracker": "https://github.com/duetosymmetry/qnm/issues",
        "Documentation": "https://qnm.readthedocs.io/",
        "Source Code": "https://github.com/duetosymmetry/qnm",
    },
    packages=setuptools.find_packages(),
    data_files = [("", ["LICENSE"])],
    package_data={'qnm':['schwarzschild/data/*']}, # TODO
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'tqdm',
        'pathlib2 ; python_version < \'3.4\'',
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)
