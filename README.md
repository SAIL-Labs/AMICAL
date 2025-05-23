<a href="https://github.com/SAIL-Labs/AMICAL">
<img src="https://raw.githubusercontent.com/SAIL-Labs/AMICAL/main/doc/Figures/amical_logo.png" width="300"></a>

(**A**perture **M**asking **I**nterferometry **C**alibration and **A**nalysis
**L**ibrary)

[![PyPI](https://img.shields.io/pypi/v/amical.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/amical/)
![Licence](https://img.shields.io/github/license/SAIL-Labs/AMICAL)

![CI](https://github.com/SAIL-Labs/AMICAL/actions/workflows/ci.yml/badge.svg)
[![CI (bleeding edge)](https://github.com/SAIL-Labs/AMICAL/actions/workflows/bleeding-edge.yaml/badge.svg)](https://github.com/SAIL-Labs/AMICAL/actions/workflows/bleeding-edge.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SAIL-Labs/AMICAL/main.svg)](https://results.pre-commit.ci/latest/github/SAIL-Labs/AMICAL/main)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

## Installation

```shell
$ python -m pip install amical
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
[example_analysis.py](https://github.com/SAIL-Labs/AMICAL/blob/main/doc/example_analysis.py) for details).

## Getting started

Looking for a quickstart into AMICAL? You can go through our **[tutorial](https://github.com/SAIL-Labs/AMICAL/blob/main/doc/tutorial.md)** explaining
how to use its different features.

You can also have a look to the example scripts
made for
[NIRISS](https://github.com/SAIL-Labs/AMICAL/blob/main/doc/example_NIRISS.py)
and
[SPHERE](https://github.com/SAIL-Labs/AMICAL/blob/main/doc/example_NIRISS.py)
or get details about the CANDID/Pymask uses with
[example_analysis.py](https://github.com/SAIL-Labs/AMICAL/blob/main/doc/example_analysis.py).

⚡ Last updates (08/2022) : New example script for IFS-SPHERE data is now available [here](https://github.com/SAIL-Labs/AMICAL/blob/main/doc/example_IFS.py).

## Use policy and reference publication

If you use AMICAL in a publication, we encourage you to properly cite the
reference paper published during the 2020 SPIE conference: [The James Webb Space
Telescope aperture masking
interferometer](https://ui.adsabs.harvard.edu/abs/2020SPIE11446E..11S/abstract).
The library explanation is part of a broader description of the interferometric
mode of NIRISS, so feel free to have a look at the exciting possibilities of
AMI!

## Acknowledgements

This work is mainly a modern Python translation of the very well known (and old)
IDL pipeline used to process and analyze Sparse Aperture Masking data. This
pipeline, called "Sydney code", was developed by a lot of people over many
years. Credit goes to the major developers, including Peter Tuthill, Mike
Ireland and John Monnier. Many forks exist across the web and the last IDL
version can be found [here](https://github.com/AnthonyCheetham/idl_masking).
