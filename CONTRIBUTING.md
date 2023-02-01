## Contributing

Bug reports and contributions are always welcome ! If you wish to contribute a
patch, please fork the project repo, create a branch on your copy, and open a
pull request (PR) here when you're done.

To install all the necessary tooling for testing and validating
your code locally, run

```shell
$ python -m pip install -r test_requirements.txt
$ python -m pip install -r typecheck_requirements.txt
```
### Fixing or adding code

We use the [pytest](https://pytest.org) framework to test AMICAL. Test files are
located in `amical/tests`, and some sample data can be found in
`amical/tests/data`.

Ideally, when fixing a bug or adding a feature, we advise you follow the
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development)
(TDD) workflow. In short, you should start by adding a failing test showing what
doesn't work (or how what's missing _should_ work), then patch the code until
your new test pass, and finally refactor for code quality if needed.

### Code formatting

The code format is validated and automatically fixed via the
[pre-commit](https://pre-commit.com) framework, most notably running
[black](https://black.readthedocs.io/en/stable/), and
[flake8](https://flake8.pycqa.org/en/latest/).

We recommend you install pre-commit on your working copy of the project, so
validation/correction is performed at checkin time (on `git commit` invokation).
If for any reason you cannot, and do not wish to use pre-commit locally, the
validation will be performed automatically by the
[pre-commit.ci](https://pre-commit.ci) bot when you open a PR. Some gotchas may
be reported by CI that cannot be autofixed, most likely by flake8.
