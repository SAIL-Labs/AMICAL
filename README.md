<a href="https://github.com/SAIL-Labs/AMICAL">
<img src="https://raw.githubusercontent.com/SAIL-Labs/AMICAL/main/docs/Figures/amical_logo.png" width="300"></a>

**A**perture **M**asking **I**nterferometry **C**alibration and **A**nalysis **L**ibrary

[![PyPI](https://img.shields.io/pypi/v/amical.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/amical/)
![Licence](https://img.shields.io/github/license/SAIL-Labs/AMICAL)

![CI](https://github.com/SAIL-Labs/AMICAL/actions/workflows/ci.yml/badge.svg)
[![CI (bleeding edge)](https://github.com/SAIL-Labs/AMICAL/actions/workflows/bleeding-edge.yaml/badge.svg)](https://github.com/SAIL-Labs/AMICAL/actions/workflows/bleeding-edge.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SAIL-Labs/AMICAL/main.svg)](https://results.pre-commit.ci/latest/github/SAIL-Labs/AMICAL/main)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

```shell
python -m pip install amical
```

AMICAL cleans aperture masking data, extracts and calibrates the visibilities
and closure phases, and saves them as OIFITS files. To analyse those files
(binary searches, contrast limits, model fitting), we recommend
[virgil](https://github.com/benjaminpope/virgil): `pip install amical[virgil]`
(Python >= 3.11). The former CANDID and Pymask functions (`amical.candid_grid`,
`amical.pymask_mcmc`, ...) remain as a legacy API computed with virgil.

See [the documentation](https://sail-labs.github.io/AMICAL/) for more information.
