import numpy as np
import pytest
from astropy.io import fits
from matplotlib import pyplot as plt

from amical import tools
from amical.get_infos_obs import get_ifu_table, get_mask


def test_find_max():
    img_size = 80  # Same size as NIRISS images
    img = np.random.random((img_size, img_size))
    xmax, ymax = np.random.randint(0, high=img_size, size=2)
    img[ymax, xmax] = img.max() * 3 + 1  # Add max pixel at pre-determined location

    center_pos = tools.find_max(img, filtmed=False)

    assert center_pos == (xmax, ymax)


def test_crop_max():
    img_size = 80  # Same size as NIRISS images
    img = np.random.random((img_size, img_size))
    xmax, ymax = np.random.randint(0, high=img_size, size=2)
    img[ymax, xmax] = img.max() * 3 + 1  # Add max pixel at pre-determined location

    # Pre-calculate expected max size
    isz_max = (
        2 * np.min([xmax, img.shape[1] - xmax - 1, ymax, img.shape[0] - ymax - 1]) + 1
    )
    isz_too_big = isz_max + 1

    # Using full message because we also check the suggested size
    size_msg = (
        f"The specified cropped image size, {isz_too_big}, is greater than the distance"
        " to the PSF center in at least one dimension. The max size for this image is"
        f" {isz_max}"
    )
    with pytest.raises(ValueError, match=size_msg):
        # Above max size should raise the error
        tools.crop_max(img, isz_too_big, filtmed=False)

    # Setting filtmed=False because the simple image has only one pixe > 1
    img_cropped, center_pos = tools.crop_max(img, isz_max, filtmed=False)

    assert center_pos == (xmax, ymax)
    assert img_cropped.shape[0] == isz_max
    assert img_cropped.shape[0] == img_cropped.shape[1]


def test_SPHERE_parang(global_datadir):
    with fits.open(global_datadir / "hdr_sphere.fits") as hdu:
        hdr = hdu[0].header
    n_ps = 1
    pa = tools.sphere_parang(hdr, n_dit_ifs=n_ps)
    true_pa = 109  # Human value
    assert pa == pytest.approx(true_pa, 0.01)


def test_NIRISS_parang(global_datadir):
    with fits.open(global_datadir / "hdr_niriss_mirage.fits") as hdu:
        hdr = hdu["SCI"].header
    pa = tools.niriss_parang(hdr)
    true_pa = 157.9079  # Human value
    assert pa == pytest.approx(true_pa, 0.01)


def test_compute_pa_sphere(global_datadir):
    with fits.open(global_datadir / "hdr_sphere.fits") as hdul:
        hdr = hdul[0].header
    n_ps = 1
    pa = tools.compute_pa(hdr, n_ps)
    true_pa = 109  # Human value
    assert pa == pytest.approx(true_pa, 0.01)


def test_compute_pa_niriss(global_datadir):
    with fits.open(global_datadir / "hdr_niriss_mirage.fits") as hdul:
        hdr = hdul[0].header
        sci_hdr = hdul["SCI"].header
        n_ps = hdul["SCI"].data.shape[-1]
    pa = tools.compute_pa(hdr, n_ps, sci_hdr=sci_hdr)
    true_pa = 157.9079  # Human value
    assert pa == pytest.approx(true_pa, 0.01)


def test_NIRISS_parang_amisim():
    # ami_sim file has no SCI
    hdr = None
    with pytest.warns(RuntimeWarning) as record:
        pa = tools.niriss_parang(hdr)
    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "No SCI header for NIRISS. No PA correction will be applied."
    )
    assert pa == 0.0


def test_compute_pa_niriss_amisim(global_datadir):
    with fits.open(global_datadir / "hdr_niriss_amisim.fits") as hdul:
        hdr = hdul[0].header
        n_ps = hdul[0].data.shape[-1]
    with pytest.warns(RuntimeWarning) as record:
        pa = tools.compute_pa(hdr, n_ps)
    assert len(record) == 1
    assert pa == 0.0


@pytest.mark.usefixtures("close_figures")
@pytest.mark.parametrize("list_index_ifu", [[0], [0, 10], [0, 1, 2]])
@pytest.mark.parametrize("filtname", ["YJ", "YH"])
def test_get_table_ifu(list_index_ifu, filtname):
    wave = get_ifu_table(list_index_ifu, filtname=filtname, display=True)
    if len(list_index_ifu) == 1:
        assert len(wave) == len(list_index_ifu)
    elif len(list_index_ifu) == 2:
        assert len(wave) == 10
    elif len(list_index_ifu) == 3:
        assert len(wave) == len(list_index_ifu)
    assert isinstance(wave, np.ndarray)
    assert plt.gcf().number == 1


def test_get_table_ifu_error():
    with pytest.raises(KeyError):
        get_ifu_table([0], instrument="fake")


@pytest.mark.parametrize("band", ["K", "L"])
@pytest.mark.parametrize("mask_name,n_holes", [("g7", 7), ("g9", 9), ("g23", 23)])
def test_get_mask_eris(mask_name, n_holes, band):
    xycoords = get_mask("ERIS", f"{mask_name}_{band}")
    assert xycoords.shape == (n_holes, 2)


def test_get_mask_eris_unknown():
    assert get_mask("ERIS", "g7_fake") is None


def test_ERIS_parang():
    hdr = fits.Header()
    hdr["HIERARCH ESO ADA PUPILPOS"] = 10.0
    hdr["HIERARCH ESO TEL PARANG START"] = 20.0
    hdr["HIERARCH ESO TEL PARANG END"] = 30.0
    pa = tools.eris_parang(hdr, n_dit=3)
    np.testing.assert_allclose(pa, [-8.0, -13.0, -18.0])


def test_wtmn_inverse_variance():
    """The calibrator average must favour the precise file, not the noisy one."""
    values = np.array([[1.0], [0.0]])
    errors = np.array([[0.01], [1.0]])
    mn, _ = tools.wtmn(values, errors)
    assert mn[0] == pytest.approx(1.0, abs=1e-3)


def test_cov2cor():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(10, 5))
    cov = samples.T @ samples

    cor, sigma = tools.cov2cor(cov)

    expected = np.empty_like(cov)
    for i in range(5):
        for j in range(5):
            expected[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    np.testing.assert_array_equal(cor, expected)
    np.testing.assert_array_equal(sigma, np.sqrt(np.diag(cov)))
    np.testing.assert_allclose(np.diag(cor), 1.0)


def test_cov2cor_negative_diagonal():
    cov = np.eye(4)
    cov[2, 2] = -1.0
    with pytest.raises(ValueError, match=r"diagonal cov\[2,2\]=-1.000000e\+00"):
        tools.cov2cor(cov)
