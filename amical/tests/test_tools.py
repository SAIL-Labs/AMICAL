import numpy as np
import pytest
from astropy.io import fits
from matplotlib import pyplot as plt

from amical import tools
from amical.get_infos_obs import get_ifu_table


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
