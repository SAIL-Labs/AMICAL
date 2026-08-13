# Contributing

Bug reports and contributions are always welcome ! If you wish to contribute a
patch, please fork the project repo, create a branch on your copy, and open a
pull request (PR) here when you're done.

## Installing AMICAL for development

The first step to install AMICAL for development is to clone the repository and `cd` into it:

```bash
git clone https://github.com/SAIL-Labs/AMICAL.git
cd AMICAL
```

You can install the package in editable mode with the development dependencies using Pip (`python -m pip install -U -e . --group dev`), but we recommend using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The next subsections discuss how to perform various development tasks.
If you installed with pip, simply remove `uv run` from the commands.

## Running the tests

AMICAL uses [pytest](https://pytest.org/) for testing.
Test files are located in `amical/tests`, and some sample data can be found in `amical/tests/data`.
To test your local installation, you can run the tests with

```bash
uv run pytest
```

## Building the documentation

AMICAL uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme for its documentation.
You should then be able to build the docs with:

```bash
uv run mkdocs build
```

The resulting files are then in the `site/` directory.
To get the docs served on a local server and reloaded as you edit, use

```bash
uv run mkdocs serve
```

## Fixing or adding code

Ideally, when fixing a bug or adding a feature, we advise you follow the
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development)
(TDD) workflow. In short, you should start by adding a failing test showing what
doesn't work (or how what's missing _should_ work), then patch the code until
your new test pass, and finally refactor for code quality if needed.

You can then [open a pull request](https://github.com/SAIL-Labs/AMICAL/compare) on the main AMICAL repository.

### Code formatting

The code format is validated and automatically fixed via the
[pre-commit](https://pre-commit.com) framework, most notably running
[ruff](https://docs.astral.sh/ruff/).
Pre-commit is installed via the `dev` dependency group.
You then need to install the pre-commit git hooks with:

```bash
uv run pre-commit install
```

Pre-commit will now run on `git commit` invokation.
If for any reason you cannot, and do not wish to use pre-commit locally, the
validation will be performed automatically by the
[pre-commit.ci](https://pre-commit.ci) bot when you open a PR.
