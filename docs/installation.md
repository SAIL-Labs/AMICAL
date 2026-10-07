# Installation

## From PyPI

The latest stable version of AMICAL can be installed with `pip` from PyPI:

```bash
python -m pip install amical
```

## From GitHub

To install the latest `main` branch directly from GitHub, use

```bash
python -m pip install git+https://github.com/SAIL-Labs/AMICAL.git
```

## Analysis with virgil

AMICAL produces calibrated OIFITS files. To analyse them, and to use the legacy
`amical.candid_*` and `amical.pymask_*` functions, install the optional
[virgil](https://github.com/benjaminpope/virgil) dependency:

```bash
python -m pip install "amical[virgil]"
```

virgil needs Python >= 3.11 and JAX. It can also be installed on its own (in
the same or a separate environment) with `python -m pip install virgil-astro`.
Note that the distribution is called `virgil-astro`: `pip install virgil`
installs an unrelated package.

## Contributing

If you wish to contribute to AMICAL, see the instructions in [CONTRIBUTING.md](https://github.com/SAIL-Labs/AMICAL/blob/main/CONTRIBUTING.md).
