"""Frozen-reference cases for the extraction regression test.

The reference outputs in ``tests/data/reference_outputs.npz`` were produced by
running this module as a script with the *original* (loop-based) AMICAL
implementation (``SAIL-Labs/AMICAL`` at commit 37945e3)::

    python tests/reference_cases.py [case ...]

so that later performance work can be checked against them.  Only regenerate a
case when its numerical output is *meant* to change, and say so in the pull
request.  Large arrays are subsampled (see ``_compact``) to keep the file small.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

DATADIR = Path(__file__).parent / "data"
REFERENCE_FILE = DATADIR / "reference_outputs.npz"

# Scalar and per-observable quantities from the extract_bs result.
RESULT_KEYS = ["u", "v", "vis2", "e_vis2", "cp", "e_cp", "bl", "bl_cp"]
# Arrays of shape (n_frames, ...): only a few frames are stored.
FRAME_KEYS = ["bs_arr", "v2_arr", "cvis_arr", "cp_arr"]
# Everything else stored in ``bs.matrix``.
MATRIX_KEYS = [
    "bs",
    "v2_cov",
    "bs_cov",
    "avar",
    "err_avar",
    "cp_cov",
    "bs_var",
    "bs_v2_cov",
    "v2_cor",
    "phs_v2corr",
]
MAX_STORED = 2048  # entries kept from any one large array


def synthetic_g18_cube(n_frames=12, npix=96, seed=18):
    """Deterministic VAMPIRES-like interferogram cube for the g18 mask.

    Each frame is a Gaussian envelope times the fringe pattern of the g18
    mask, with seeded random hole pistons (seeing) and Gaussian noise.
    """
    from amical.get_infos_obs import get_mask, get_pixel_size, get_wavelength

    xy = get_mask("VAMPIRES", "g18")  # hole positions [m]
    wl = get_wavelength("VAMPIRES", "750-50")[0]  # [m]
    pix = get_pixel_size("VAMPIRES")  # [rad]

    rng = np.random.default_rng(seed)
    y, x = (np.mgrid[:npix, :npix] - npix // 2) * pix
    env = np.exp(-((x / pix) ** 2 + (y / pix) ** 2) / (2 * 12.0**2))
    cube = np.empty((n_frames, npix, npix))
    for i in range(n_frames):
        piston = rng.normal(0, 0.6, len(xy))
        field = np.zeros((npix, npix), dtype=complex)
        for (xh, yh), p in zip(xy, piston, strict=True):
            field += np.exp(2j * np.pi * (xh * x + yh * y) / wl + 1j * p)
        img = 1e3 * env * np.abs(field) ** 2
        cube[i] = img + rng.normal(0, 0.01 * img.max(), img.shape)
    return cube


def _niriss_cube(n_frames=None):
    fits_file = DATADIR / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    if n_frames is not None:
        cube = cube[:n_frames]
    return cube, fits_file


def _g18_cube(tmp_dir):
    cube = synthetic_g18_cube()
    fits_file = Path(tmp_dir) / "synthetic_g18.fits"
    hdu = fits.PrimaryHDU(cube)
    hdu.header["INSTRUME"] = "VAMPIRES"
    hdu.writeto(fits_file, overwrite=True)
    return cube, fits_file


def run_case(name, tmp_dir):
    """Run ``extract_bs`` for a named reference case and return the result."""
    import amical

    common = {"targetname": "test", "display": False, "verbose": False}
    if name == "niriss_fft":
        cube, fits_file = _niriss_cube()
        kw = {"maskname": "g7", "fw_splodge": 0.7, "peakmethod": "fft"}
    elif name == "niriss_gauss":
        cube, fits_file = _niriss_cube()
        kw = {"maskname": "g7", "fw_splodge": 0.7, "peakmethod": "gauss"}
    elif name == "vampires_g18":
        cube, fits_file = _g18_cube(tmp_dir)
        kw = {"maskname": "g18", "filtname": "750-50", "peakmethod": "fft"}
    elif name == "niriss_multitri":
        cube, fits_file = _niriss_cube(n_frames=40)
        kw = {"maskname": "g7", "fw_splodge": 0.7, "peakmethod": "fft"}
        kw["bs_multi_tri"] = True
    elif name == "vampires_g18_multitri":
        cube, fits_file = _g18_cube(tmp_dir)
        kw = {"maskname": "g18", "filtname": "750-50", "peakmethod": "fft"}
        kw["bs_multi_tri"] = True
    else:
        raise ValueError(f"Unknown reference case {name!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return amical.extract_bs(cube, fits_file, **common, **kw)


CASES = [
    "niriss_fft",
    "niriss_gauss",
    "vampires_g18",
    "niriss_multitri",
    "vampires_g18_multitri",
]


# Multiple-triangle cases only differ in the bispectrum, so they store just
# the bispectrum-derived quantities, more sparsely.
MULTITRI_KEYS = {"cp", "e_cp", "bl_cp", "bs", "bs_cov", "cp_cov", "bs_var"}
MULTITRI_KEYS |= {"bs_v2_cov", "bs_arr", "cp_arr"}
MULTITRI_MAX_STORED = 512


def _compact(name, arr, rng, max_stored=MAX_STORED):
    """Return {key: array} entries storing ``arr`` (or a subsample of it)."""
    arr = np.asarray(arr)
    if arr.size <= max_stored:
        return {name: arr}
    flat = np.sort(rng.choice(arr.size, max_stored, replace=False))
    return {name: arr.ravel()[flat], name + "__idx": flat}


def collect(name, bs):
    """Flatten an ``extract_bs`` result into reference entries for ``name``."""
    rng = np.random.default_rng(0)
    keep, max_stored = set(RESULT_KEYS + MATRIX_KEYS + FRAME_KEYS), MAX_STORED
    if name.endswith("_multitri"):
        keep, max_stored = MULTITRI_KEYS, MULTITRI_MAX_STORED
    out = {}
    for key in RESULT_KEYS + MATRIX_KEYS + FRAME_KEYS:
        if key not in keep:
            continue
        arr = np.asarray(bs[key] if key in RESULT_KEYS else bs.matrix[key])
        if key in FRAME_KEYS:
            arr = arr[[0, len(arr) // 2, len(arr) - 1]]
        out.update(_compact(f"{name}/{key}", arr, rng, max_stored))
    return out


if __name__ == "__main__":
    import tempfile

    names = sys.argv[1:] or CASES
    entries = dict(np.load(REFERENCE_FILE)) if REFERENCE_FILE.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            entries = {k: v for k, v in entries.items() if not k.startswith(name + "/")}
            entries.update(collect(name, run_case(name, tmp)))
            print(f"{name}: done")
    np.savez_compressed(REFERENCE_FILE, **entries)
    print(f"Wrote {REFERENCE_FILE} ({REFERENCE_FILE.stat().st_size / 1e3:.0f} kB)")
