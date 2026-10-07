# AMICAL Documentation

<a href="https://github.com/SAIL-Labs/AMICAL">
<img src="Figures/amical_logo.png" width="300"></a>

**A**perture **M**asking **I**nterferometry **C**alibration and **A**nalysis **L**ibrary

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

AMICAL's job ends at (4) the calibrated **OIFITS** file. To **analyse** it
(binary searches, contrast limits, model fitting and posteriors), we recommend
[virgil](https://github.com/benjaminpope/virgil) (`pip install amical[virgil]`),
which reads AMICAL's OIFITS files directly. AMICAL no longer bundles
[CANDID](https://github.com/amerand/CANDID) and
[Pymask](https://github.com/AnthonyCheetham/pymask): `amical.candid_grid`,
`amical.pymask_mcmc` and the other functions that used them remain as a legacy
API, computed with virgil, and `amical.smartfit` is deprecated. See the
[companion search tutorial](tutorials/companion-search.ipynb) for both.

## Getting started

Looking for a quickstart into AMICAL? You can go through our **[getting started tutorial](getting-started.ipynb)** explaining
how to use its different features.

You can also have a look to the tutorials
for [NIRISS](tutorials/niriss-extraction.ipynb) and [SPHERE](tutorials/sphere-extraction.ipynb) or see how to analyse the calibrated OIFITS file in [the companion search tutorial](tutorials/companion-search.ipynb).

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

New contributions are always welcome!
If you wish to contribute to AMICAL, take a look at [CONTRIBUTING.md](https://github.com/SAIL-Labs/AMICAL/blob/main/CONTRIBUTING.md).

AMICAL core developers:

- [Anthony Soulain](https://github.com/drsoulain)
- [Clément Robert](https://github.com/neutrinoceros)
- [Thomas Vandal](https://github.com/vandalt)

IDL Masking Code:

- [Peter Tuthill](http://www.physics.usyd.edu.au/~gekko/)
- [Mike Ireland](https://github.com/mikeireland)
- [John Monnier](https://lsa.umich.edu/astro/people/core-faculty/monnier.html)

The analysis functions were originally interfaces to the following packages
(now replaced by [virgil](https://github.com/benjaminpope/virgil)):

Pymask:

- [Benjamin Pope](https://github.com/benjaminpope)
- [Anthony Cheetham](https://github.com/AnthonyCheetham)
- Based on [xara](https://github.com/fmartinache/xara) (formerly pysco), by [Frantz Martinache](https://github.com/fmartinache)

CANDID:

- [Antoine Mérand](https://github.com/amerand)
- Alexandre Gallenne
