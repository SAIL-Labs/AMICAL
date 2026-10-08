"""Regression test against frozen outputs of the original AMICAL extraction.

``tests/data/reference_outputs.npz`` was generated with the original loop-based
implementation (see ``tests/reference_cases.py``).  Vectorised or otherwise
optimised code must reproduce it to floating-point noise.
"""

import numpy as np
import pytest
from reference_cases import CASES, REFERENCE_FILE, collect, run_case

# Agreement required relative to the largest stored magnitude of each array.
# Results agree to ~1e-14 on the machine that generated the reference; the
# margin absorbs differences in BLAS/FFT/libm between platforms and numpy
# versions.  Element-wise relative agreement is not meaningful for quantities
# that pass through zero (closure phases, covariances).
TOLERANCE = 1e-10


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE_FILE) as ref:
        return dict(ref)


@pytest.fixture(scope="module", params=CASES)
def case(request, tmp_path_factory):
    name = request.param
    return name, run_case(name, tmp_path_factory.mktemp(name))


def test_reference_outputs(case, reference):
    name, bs = case
    actual = collect(name, bs)
    keys = [k for k in reference if k.startswith(name + "/") and "__idx" not in k]
    assert keys, f"No reference entries for case {name}"
    for key in keys:
        ref = reference[key]
        if key + "__idx" in reference:
            # ``collect`` subsamples with the same seeded indices.
            np.testing.assert_array_equal(
                actual[key + "__idx"], reference[key + "__idx"]
            )
        got = actual[key]
        assert got.shape == ref.shape, key
        scale = np.max(np.abs(ref))
        np.testing.assert_allclose(
            got, ref, rtol=0, atol=TOLERANCE * scale, err_msg=key
        )
