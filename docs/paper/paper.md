---
title: 'qnm: A Python package for calculating Kerr quasinormal modes, separation constants, and spherical-spheroidal mixing coefficients'
tags:
  - Python
  - physics
  - general relativity
  - black holes
authors:
  - name: Leo C. Stein
    orcid: 0000-0001-7559-9597
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physics and Astronomy, The University of Mississippi, University, MS 38677, USA
   index: 1
date: 20 August 2019
bibliography: paper.bib
---

# Background

Black holes can be characterized from far away by their spectroscopic
gravitational-wave ``fingerprints,'' in analogy to electromagnetic
spectroscopy of atoms, ions, and molecules.  The idea of using the
quasi-normal modes (QNMs) of black holes (BHs) for gravitational-wave
(GW) spectroscopy was first made explicit by Detweiler
[@Detweiler:1980gk].  QNMs of rotating Kerr BHs in general relativity
(GR) depend only on the mass and spin of the BH.  Thus GWs containing
QNMs can be used to infer the remnant BH properties in a binary
merger, or as a test of GR by checking the consistency between the
inspiral and ringdown portions of a GW signal
[@TheLIGOScientific:2016src; @Isi:2019aib].

For a review of QNMs see [@Berti:2009kk].  A Kerr BH's QNMs are the
homogeneous (source-free) solutions to the Teukolsky equation
[@Teukolsky:1973ha] subject to certain physical conditions.  The
Teukolsky equation can apply to different physical fields based on
their spin-weight $s$; for gravitational perturbations, we are
interested in $s=-2$ (describing the Newman-Penrose scalar $\psi_4$).
The physical conditions for a QNM are quasi-periodicity in time, of
the form $\propto e^{-i \omega t}$ with complex $\omega$; conditions
of regularity, and that the solution has waves that are only going
down the horizon and out at spatial infinity.  Separating the
radial/angular Teukolsky equations and imposing these conditions gives
an eigenvalue problem where the frequency $\omega$ and separation
constant $A$ must be found simultaneously.  This eigenvalue problem
has a countably infinite, discrete spectrum labeled by angular
harmonic numbers $(\ell, m)$ with $\ell\ge 2$ (or $\ell \ge |s|$ for
fields of other spin weight), $-\ell \le m \le +\ell$, and overtone
number $n \ge 0$.

There are several analytic techniques, e.g. [@Dolan:2009nk], to
approximate the desired complex frequency and separation constant
$(\omega_{\ell, m, n}(a), A_{\ell, m, n}(a))$ as a function of spin
parameter $0 \le a < M$ (we follow the convention of using units where
the total mass is $M=1$).  These analytic techniques are useful as
starting guesses before applying the numerical method of Leaver
[@Leaver:1985ax] for root-polishing.  Leaver's method uses Frobenius
expansions of the radial and angular Teukolsky equations to find
3-term recurrence relations that must be satisfied at a complex
frequency $\omega$ and separation constant $A$.  The recurrence
relations are made numerically stable to find so-called minimal
solutions by being turned into infinite continued fractions.  In
Leaver's approach, there are thus two ``error'' functions $E_r(\omega,
A)$ and $E_a(\omega, A)$ (each depending on $a, \ell, m, n$) which are
given as infinite continued fractions, and the goal is to find a pair
of complex numbers $(\omega, A)$ which are simultaneous roots of both
functions.  This is typically accomplished by complex root-polishing,
alternating between the radial and angular continued fractions.

A refinement of this method was put forth by
[@Cook:2014cta] (see also Appendix A of [@Hughes:1999bq]).  Instead of
solving the angular Teukolsky equation ``from the endpoint'' using
Leaver's approach, one can use a spectral expansion with a good choice
of basis functions.  The solutions to the angular problem are the
*spin-weighted spheroidal harmonics*, and the appropriate spectral
basis are the spin-weighted *spherical* harmonics.  This expansion is
written as (spheroidal on the left, sphericals on the right):

$${}_s Y_{\ell m}(\theta, \phi; a\omega) = {\sum_{\ell'=\ell_{\min} (s,m)}^{\ell_{\max}}} C_{\ell' \ell m}(a\omega)\ {}_s Y_{\ell' m}(\theta, \phi) \,,$$

where $\ell_{\min} = \max(|m|, |s|)$, and the coefficients
$C_{\ell' \ell m}(a\omega)$ are called the spherical-spheroidal mixing
coefficients (we follow the conventions of [@Cook:2014cta], but
compare [@Berti:2014fga]).  When recast in this spectral form, the
angular equation becomes very easy to solve via standard matrix
eigenvector routines, see [@Cook:2014cta] for details.
If one picks values for $(s, \ell, m, a, \omega)$, then
the separation constant $A(a\omega)$ is returned as an eigenvalue, and
a vector of mixing coefficients $C_{\ell' \ell m}(a\omega)$ are
returned as an eigenvector.  From this new point of view there is now
only one error function to root-polish,
$E_r(\omega) = E_r(\omega, A(\omega))$ where the angular separation
constant is found from the matrix method at any value of $\omega$.
Polishing roots of $E_r$ proceeds via any standard 2-dimensional
root-finding or optimization method.

The main advantage of the spectral approach is rapid convergence, and
getting the spherical-spheroidal mixing coefficients ``for free''
since they are found in the process of solving the spectral angular
eigenvalue problem.

# Summary

``qnm`` is an open-source Python package for computing the Kerr QNM
frequencies, angular separation constants, and spherical-spheroidal
mixing coefficients, for given values of $(\ell, m, n)$ and spin $a$.
There are several QNM codes available, but some
([London](https://github.com/llondon6/kerr_public)) implement either
analytic fitting formulae (which only exist for a range of $s, \ell,
m, n$) or interpolation from tabulated data (so the user can not
root-polish); others
([Berti](https://pages.jh.edu/~eberti2/ringdown/)) are in proprietary
languages such as Mathematica.  We are not aware of any packages that
provide spherical-spheroidal mixing coefficients, which are necessary
for multi-mode ringdown GW modeling.

The ``qnm`` package includes a Leaver solver with the Cook-Zalutskiy
spectral approach to the angular sector, thus providing mixing
coefficients.  We also include a caching mechanism to avoid repeating
calculations.  When the user wants to solve at a new value of $a$, the
cached data is used to interpolate a good initial guess for
root-polishing.  We provide a large cache of low $\ell, m, n$ modes so
the user can start interpolating right away, and this precomputed
cache can be downloaded and installed with a single function call.  We
have adapted the core algorithms so that ``numba`` [@Numba] can
just-in-time compile them to optimized, machine-speed code.  We rely
on ``numpy`` [@NumPy] for common operations such as solving the
angular eigenvalue problem, and we rely on ``scipy`` [@SciPy] for
two-dimensional root-polishing, and interpolating from the cache
before root-polishing.

This package should enable researchers to perform ringdown modeling of
gravitational-wave data in Python, without having to interpolate into
precomputed tables or write their own Leaver solver.  The author and
collaborators are already using this package for multiple active
research projects.  By creating a self-documented, open-source code,
we hope to alleviate the high frequency of re-implemenation of
Leaver's method, and instead focus efforts on making a single robust,
fast, high-precision, and easy-to-use code for the whole community.
In the future, this code can be extended to incorporate new features
(like special handling of algebraically special modes) or to apply to
more general BH solutions (e.g. solving for QNMs of Kerr-Newman or
Kerr-de Sitter).

Development of ``qnm`` is hosted on
[GitHub](https://github.com/duetosymmetry/qnm) and distrubted through
[PyPI](https://pypi.org/project/qnm/); it can be installed with the
single command ``pip install qnm``.  Documentation is automatically
built on [Read the Docs](https://qnm.readthedocs.io/), and can be
accessed interactively via Python docstrings.  The ``qnm`` package is
part of the [Black Hole Perturbation Theory
Toolkit](https://bhptoolkit.org/).

# Acknowledgements

We acknowledge E Berti and GB Cook for helpful correspondence, and E
Berti for making testing data available.  We further acknowledge M
Giesler, I Hawke, L Magaña Zertuche, and V Varma for
contributions/testing/feedback/suggestions.

# References
