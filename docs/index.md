<a href="https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL">
<img src="https://raw.githubusercontent.com/SydneyAstrophotonicInstrumentationLab/AMICAL/master/doc/Figures/amical_logo.png" width="300"></a>

(**A**perture **M**asking **I**nterferometry **C**alibration and **A**nalysis
**L**ibrary)

[![PyPI](https://img.shields.io/pypi/v/amical)](https://pypi.org/project/amical/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/amical)](https://pypi.org/project/amical/)
![Licence](https://img.shields.io/github/license/SydneyAstrophotonicInstrumentationLab/AMICAL)

![CI](https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/actions/workflows/ci.yml/badge.svg)
[![CI (bleeding edge)](https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/actions/workflows/bleeding-edge.yaml/badge.svg)](https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/actions/workflows/bleeding-edge.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SydneyAstrophotonicInstrumentationLab/AMICAL/master.svg)](https://results.pre-commit.ci/latest/github/SydneyAstrophotonicInstrumentationLab/AMICAL/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

```shell
$ python3 -m pip install amical
```

## What can AMICAL do for you ?

AMICAL is developed to provide an easy-to-use solution to process
**A**perture **M**asking **I**nterferometry (AMI) data from major existing
facilities:
[NIRISS](https://jwst-docs.stsci.edu/near-infrared-imager-and-slitless-spectrograph)
on the JWST (first scientific interferometer operating in space),
[SPHERE](https://www.eso.org/sci/facilities/paranal/instruments/sphere.html) and
[VISIR](https://www.eso.org/sci/facilities/paranal/instruments/visir.html) from
the European Very Large Telescope (VLT) and
[VAMPIRES](https://www.naoj.org/Projects/SCEXAO/scexaoWEB/030openuse.web/040vampires.web/indexm.html)
from SUBARU telescope (and more to come).

We focused our efforts to propose a user-friendly interface, though different
sub-classes allowing to (1) **Clean** the reduced datacube from the standard
instrument pipelines, (2) **Extract** the interferometrical quantities
(visibilities and closure phases) using a Fourier sampling approach and (3)
**Calibrate** those quantities to remove the instrumental biases.

In addition (4), we include two external packages called
[CANDID](https://github.com/amerand/CANDID) and
[Pymask](https://github.com/AnthonyCheetham/pymask) to **analyse** the final
outputs obtained from a binary-like sources (star-star or star-planet). We
interfaced these stand-alone packages with AMICAL to quickly estimate our
scientific results (e.g., separation, position angle, contrast ratio, contrast
limits, etc.) using different approaches (chi2 grid, MCMC, see
[example_analysis.py](https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/blob/master/doc/example_analysis.py) for details).

## Getting started

Looking for a quickstart into AMICAL? You can go through our **[tutorial](tutorial.md)** explaining
how to use its different features.

You can also have a look to the example scripts
made for [NIRISS](notebooks/example_NIRISS.ipynb) and [SPHERE](https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/blob/master/doc/example_NIRISS.py) or get details about the CANDID/Pymask uses with [example_analysis.py](notebooks/example_analysis.py).

## Use policy and reference publication

If you use AMICAL in a publication, we encourage you to properly cite the
reference paper published during the 2020 SPIE conference: 

[The James Webb Space
Telescope aperture masking
interferometer](https://ui.adsabs.harvard.edu/abs/2020SPIE11446E..11S/abstract)


with BibTex:

```BibTex
@INPROCEEDINGS{2020SPIE11446E..11S,
       author = {{Soulain}, A. and {Sivaramakrishnan}, A. and {Tuthill}, P. and {Thatte}, D. and {Volk}, K. and {Cooper}, R. and {Albert}, L. and {Artigau}, {\'E}. and {Cook}, N. and {Doyon}, R. and {Johnstone}, D. and {Lafreni{\`e}re}, D. and {Martel}, A.},
        title = "{The James Webb Space Telescope aperture masking interferometer}",
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics},
    booktitle = {Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series},
         year = 2020,
       series = {Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series},
       volume = {11446},
        month = dec,
          eid = {1144611},
        pages = {1144611},
          doi = {10.1117/12.2560804},
archivePrefix = {arXiv},
       eprint = {2201.01524},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020SPIE11446E..11S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Acknowledgements

This work is mainly a modern Python translation of the very well known (and old)
IDL pipeline used to process and analyze Sparse Aperture Masking data. This
pipeline, called "Sydney code", was developed by a lot of people over many
years. Credit goes to the major developers, including Peter Tuthill, Mike
Ireland and John Monnier. Many forks exist across the web and the last IDL
version can be found [here](https://github.com/AnthonyCheetham/idl_masking).

## Contributors

AMICAL core developers:

- [Anthony Soulain](https://github.com/drsoulain)
- [Clément Robert](https://github.com/neutrinoceros)
- [Thomas Vandal](https://github.com/vandalt)

IDL Masking Code: 

- [Peter Tuthill](http://www.physics.usyd.edu.au/~gekko/)
- [Mike Ireland](https://github.com/mikeireland)
- [John Monnier](https://lsa.umich.edu/astro/people/core-faculty/monnier.html)

Pymask:

- [Benjamin Pope](https://github.com/benjaminpope)
- [Anthony Cheetham](https://github.com/AnthonyCheetham)
- Based on pysco, now [xara](https://github.com/fmartinache/xara), by [Frantz Martinache](https://github.com/fmartinache)

CANDID:

- [Antoine Mérand](https://github.com/amerand)
- A Galenne