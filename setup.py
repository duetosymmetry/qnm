'''
Standard setup.py to upload the code on pypi.

    python setup.py sdist bdist_wheel
    twine upload dist/*
'''
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

import sys
sys.path.append("qnm")

from _version import __version__

setuptools.setup(
    name="qnm",
    version=__version__,
    author="Leo C. Stein",
    author_email="leo.stein@gmail.com",
    description="Calculate quasinormal modes of Kerr black holes.",
    keywords='black-holes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duetosymmetry/qnm/",
    packages=setuptools.find_packages(),
    package_data={'qnm':['schwarzschild/data/*']}, # TODO
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
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
