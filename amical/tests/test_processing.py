import numpy as np
import pytest
from matplotlib import pyplot as plt

import amical
from amical.data_processing import clean_data
from amical.data_processing import sky_correction


@pytest.fixture()
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def test_sky_out_image():
    img_dim = 80
    img = np.ones((img_dim, img_dim))

    # Inner radius beyond image corners
    r1 = np.sqrt(2 * (img_dim / 2) ** 2) + 2

    with pytest.warns(
        RuntimeWarning,
        match="Background not computed, likely because specified radius is out of bounds",
    ):
        img_bg, bg = sky_correction(img, r1=r1)

    assert np.all(img_bg == img)
    assert np.all(bg == 0)


@pytest.mark.usefixtures("close_figures")
def test_sky_inner_only():
    img_dim = 80
    img = np.ones((img_dim, img_dim))

    # Inner radius beyond image corners
    r1 = np.sqrt(2 * (img_dim / 2) ** 2) - 10
    dr = 100

    with pytest.warns(
        RuntimeWarning,
        match="The outer radius is out of the image, using everything beyond r1 as background",
    ):
        sky_correction(img, r1=r1, dr=dr)


@pytest.mark.usefixtures("close_figures")
def test_clean_data_none_kwargs():
    # Test clean_data when the "main" kwargs are set to None
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    # sky=True raises a warning by default because required kwargs are None
    with pytest.warns(
        RuntimeWarning,
        match="sky is set to True, but .* set to None. Skipping sky correction",
    ):
        cube_clean_sky = clean_data(data)

    # apod=True raises a warning by default because required kwargs are None
    with pytest.warns(
        RuntimeWarning,
        match="apod is set to True, but window is None. Skipping apodisation",
    ):
        cube_clean_apod = clean_data(data, sky=False)

    cube_clean = clean_data(data, sky=False, apod=False)

    assert (data == cube_clean).all()
    assert np.logical_and(
        cube_clean == cube_clean_apod, cube_clean == cube_clean_sky
    ).all()


@pytest.mark.usefixtures("close_figures")
def test_clean(global_datadir):
    fits_file = global_datadir / "test.fits"

    clean_param = {
        "isz": 79,
        "r1": 35,
        "dr": 2,
        "apod": True,
        "window": 65,
        "f_kernel": 3,
        "add_bad": [[39, 39]],
    }

    cube_clean = amical.select_clean_data(
        fits_file, clip=True, **clean_param, display=True
    )

    amical.show_clean_params(fits_file, **clean_param)
    amical.show_clean_params(fits_file, **clean_param, remove_bad=False)

    im1 = amical.data_processing._apply_patch_ghost(
        cube_clean, 40, 40, radius=20, dx=3, dy=3, method="zero"
    )
    im2 = amical.data_processing._apply_patch_ghost(
        cube_clean, 40, 40, radius=20, dx=3, dy=3, method="bg"
    )

    assert type(cube_clean) == np.ndarray
    assert plt.gcf().number == 4
    assert im1.shape == cube_clean.shape
    assert im2.shape == cube_clean.shape
    plt.close("all")
