import numpy as np
import pytest
from matplotlib import pyplot as plt

import amical
from amical.data_processing import clean_data
from amical.data_processing import fix_bad_pixels
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

    cube_clean = amical.select_clean_data(fits_file, clip=True, **clean_param)

    amical.show_clean_params(fits_file, **clean_param)
    amical.show_clean_params(fits_file, **clean_param, remove_bad=False)

    im1 = amical.data_processing._apply_patch_ghost(
        cube_clean, 40, 40, radius=20, dx=3, dy=3, method="zero"
    )
    im2 = amical.data_processing._apply_patch_ghost(
        cube_clean, 40, 40, radius=20, dx=3, dy=3, method="bg"
    )

    assert type(cube_clean) == np.ndarray
    assert im1.shape == cube_clean.shape
    assert im2.shape == cube_clean.shape


def test_fix_bad_pixel_no_bad():
    img_dim = 80
    data = np.random.random((img_dim, img_dim))

    no_bpix = fix_bad_pixels(data, np.zeros_like(data, dtype=bool))

    assert np.all(data == no_bpix)


def test_fix_one_bad_pixel():

    img_dim = 80
    data = np.random.random((img_dim, img_dim))

    bad_ind = tuple(np.random.randint(0, high=img_dim, size=2))
    data[bad_ind] = 1e5

    bad_map = np.zeros_like(data, dtype=bool)
    bad_map[bad_ind] = 1

    no_bpix = fix_bad_pixels(data, bad_map)

    assert no_bpix[bad_map] != data[bad_map]
    assert np.all(no_bpix[~bad_map] == data[~bad_map])
    assert 0.0 <= no_bpix[bad_ind] <= 1.0  # Because test data is random U(0, 1)


@pytest.mark.usefixtures("close_figures")
def test_clean_data_no_bmap_add_bad():
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_ind = tuple(np.random.randint(0, high=img_dim, size=2))
    bad_ind_3d = (slice(None),) + bad_ind
    add_bad = [bad_ind[::-1]]

    # Put value out of data bounds (0, 1) to make sure corrected != original
    data[bad_ind_3d] = 1e5

    # Test case with non-emtpy add_bad but empty add_bad
    cleaned = clean_data(data, sky=False, apod=False, add_bad=add_bad)
    nobpix_list = [
        fix_bad_pixels(img, bad_map=np.zeros((img_dim, img_dim)), add_bad=add_bad)
        for img in data
    ]

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[bad_ind_3d] != data[bad_ind_3d])


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_2d():
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_map = np.zeros((img_dim, img_dim), dtype=bool)
    bad_map[tuple(np.random.randint(0, high=img_dim, size=2))] = 1

    data[:, bad_map] = 1e5

    # Test case with non-emtpy add_bad but empty add_bad
    cleaned = clean_data(data, sky=False, apod=False, bad_map=bad_map)
    nobpix_list = [fix_bad_pixels(img, bad_map=bad_map) for img in data]

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[:, bad_map] != data[:, bad_map])
    assert np.all(cleaned[:, ~bad_map] == data[:, ~bad_map])


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_add_bad_2d():
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    new_bad_ind = tuple(np.random.randint(0, high=img_dim, size=2))
    add_bad = [new_bad_ind[::-1]]

    bad_map = np.zeros((img_dim, img_dim), dtype=bool)
    bad_map[tuple(np.random.randint(0, high=img_dim, size=2))] = 1

    # Test case with non-emtpy add_bad but empty add_bad
    cleaned = clean_data(data, sky=False, apod=False, bad_map=bad_map, add_bad=add_bad)
    nobpix_list = [
        fix_bad_pixels(img, bad_map=bad_map, add_bad=add_bad) for img in data
    ]

    full_bad_map = bad_map.copy()
    full_bad_map[new_bad_ind] = True

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[:, full_bad_map] != data[:, full_bad_map])
    assert np.all(cleaned[:, ~full_bad_map] == data[:, ~full_bad_map])


def test_clean_data_bmap_2d_shape():
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_map = np.zeros((img_dim, img_dim - 1), dtype=bool)

    with pytest.raises(
        ValueError, match="bad_map should have the same shape as a frame"
    ):
        clean_data(data, sky=False, apod=False, bad_map=bad_map)
