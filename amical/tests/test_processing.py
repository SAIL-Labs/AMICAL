import numpy as np
import pytest
from matplotlib import pyplot as plt

import amical
from amical.data_processing import _get_3d_bad_pixels
from amical.data_processing import _get_ring_mask
from amical.data_processing import clean_data
from amical.data_processing import fix_bad_pixels
from amical.data_processing import sky_correction
from amical.tools import crop_max


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

    # Outer radius beyond image corners
    r1 = np.sqrt(2 * (img_dim / 2) ** 2) - 10
    dr = 100

    with pytest.warns(
        RuntimeWarning,
        match="The outer radius is out of the image, using everything beyond r1 as background",
    ):
        sky_correction(img, r1=r1, dr=dr)


@pytest.mark.usefixtures("close_figures")
def test_clean_sky_out_crop():

    n_im = 5
    img_dim = 80
    data = np.ones((n_im, img_dim, img_dim))
    xmax, ymax = (40, 41)
    data[:, ymax, xmax] = data.max() * 3 + 1  # Add max pixel at pre-determined location

    isz = 67

    r1 = int(np.sqrt(2 * (isz / 2) ** 2) + 2)

    clean_cube = clean_data(data, isz=isz, r1=r1, dr=2, f_kernel=None, apod=False)

    # Make sure did not just remove everything
    assert len(clean_cube) != 0


@pytest.mark.usefixtures("close_figures")
def test_sky_dr_none():
    # Make sure dr=None does not crash
    img_dim = 80
    img = np.ones((img_dim, img_dim))

    # Outer radius beyond image corners
    r1 = np.sqrt(2 * (img_dim / 2) ** 2) - 10
    dr = None

    sky_correction(img, r1=r1, dr=dr)


@pytest.mark.usefixtures("close_figures")
def test_sky_correction_mask_all():
    # Check that full correction mask gives expected result
    img_dim = 80
    img = np.random.random((img_dim, img_dim))
    mask = img.astype(bool)

    img_sky = sky_correction(img, mask=mask)[0]

    img_corr = img + 1.01 * np.abs(img.min())
    img_corr = img_corr - np.mean(img_corr[mask])

    assert np.all(img_sky == img_corr)


def test_sky_correction_mask_zeros():
    # Check that empty correction mask gives warning and does nothing
    img_dim = 80
    img = np.random.random((img_dim, img_dim))
    mask = np.zeros_like(img, dtype=bool)

    with pytest.warns(
        RuntimeWarning,
        match="Background not computed because mask has no True values",
    ):
        img_corr = sky_correction(img, mask=mask)[0]

    assert np.all(img_corr == img)


def test_sky_ring_mask():
    img_dim = 80
    img = np.random.random((img_dim, img_dim))

    r1 = 20
    dr = 2
    mask = _get_ring_mask(r1, dr, img.shape[0])

    img_mask = sky_correction(img, mask=mask)[0]
    img_ring = sky_correction(img, r1=r1, dr=dr)[0]

    assert np.all(img_mask == img_ring)


def test_sky_correction_deprecation():
    # Check deprecation warning is raised for r1 and dr
    img_dim = 400  # Big image because default r1=100
    img = np.random.random((img_dim, img_dim))

    # FUTURE: Future AMICAL release should raise error, see sky_correction()
    msg = (
        "The default value of r1 and dr is now None. Either mask or r1 must be set"
        " explicitely. In the future, this will result in an error."
        " Setting r1=100 and dr=20"
    )
    with pytest.warns(
        PendingDeprecationWarning,
        match=msg,
    ):
        img_default = sky_correction(img)[0]

    img_corr = sky_correction(img, r1=100, dr=20)[0]

    assert np.all(img_corr == img_default)


def test_sky_correction_errors():

    img_dim = 80
    img = np.random.random((img_dim, img_dim))

    # Test both r1 and mask given
    with pytest.raises(TypeError, match="Only one of mask and r1 can be specified"):
        sky_correction(img, r1=20, mask=np.ones_like(img, dtype=bool))
    # Test dr but no r1
    with pytest.raises(TypeError, match="dr cannot be set when r1 is None"):
        sky_correction(img, dr=5, mask=np.ones_like(img, dtype=bool))
    # Test bad mask shape
    with pytest.raises(ValueError, match="mask should have the same shape as image"):
        sky_correction(img, mask=np.ones(5))


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


def test_clean_data_dr_none():
    # Test clean_data with dr=None (i.e. single sky delimiter)
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    r1 = int(np.sqrt(2 * (img_dim / 2) ** 2) - 2)  # working r1

    clean_data(data, r1=r1, apod=False)


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
    # Test add_data kwarg when no bad_map
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_ind = tuple(np.random.randint(0, high=img_dim, size=2))
    bad_ind_3d = (slice(None),) + bad_ind
    add_bad = [bad_ind[::-1]]

    # Set values out of random range to test that they did change later
    data[bad_ind_3d] = 1e5

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
    # Test combination of bad_map and add_bad in 2d
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    new_bad_ind = tuple(np.random.randint(0, high=img_dim, size=2))
    new_bad_ind_3d = (slice(None),) + new_bad_ind
    add_bad = [new_bad_ind[::-1]]

    bad_map = np.zeros((img_dim, img_dim), dtype=bool)
    bad_map[tuple(np.random.randint(0, high=img_dim, size=2))] = 1

    # Set values out of random range to test that they did change later
    data[:, bad_map] = 1e5
    data[new_bad_ind_3d] = 1e5

    cleaned = clean_data(data, sky=False, apod=False, bad_map=bad_map, add_bad=add_bad)
    nobpix_list = [
        fix_bad_pixels(img, bad_map=bad_map, add_bad=add_bad) for img in data
    ]

    full_bad_map = bad_map.copy()
    full_bad_map[new_bad_ind] = True

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[:, full_bad_map] != data[:, full_bad_map])
    assert np.all(cleaned[:, ~full_bad_map] == data[:, ~full_bad_map])


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_2d_shape():
    # Assert that bad shape raises error in 2d
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_map = np.zeros((img_dim, img_dim - 1), dtype=bool)

    with pytest.raises(
        ValueError, match="2D bad_map should have the same shape as a frame"
    ):
        clean_data(data, sky=False, apod=False, bad_map=bad_map)


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_3d():
    # Test regular bad pixel map per-frame
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_cube = np.zeros((n_im, img_dim, img_dim), dtype=bool)
    for i in range(n_im):
        bad_cube[(i, *np.random.randint(0, high=img_dim, size=2))] = 1

    # Put value out of data bounds (0, 1) to make sure corrected != original
    data[bad_cube] = 1e5

    cleaned = clean_data(data, sky=False, apod=False, bad_map=bad_cube)
    nobpix_list = [
        fix_bad_pixels(img, bad_map=bmap) for img, bmap in zip(data, bad_cube)
    ]

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[bad_cube] != data[bad_cube])
    assert np.all(cleaned[~bad_cube] == data[~bad_cube])


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_3d_shape():
    # Assert that bad shape raises error in 3d
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_map = np.zeros((n_im, img_dim, img_dim - 1), dtype=bool)

    with pytest.raises(
        ValueError, match="3D bad_map should have the same shape as data cube"
    ):
        clean_data(data, sky=False, apod=False, bad_map=bad_map)


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_3d_add_bad_2d():
    # Test combination of 3d bad pixels with add_bad common for all dimensions
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    new_bad_ind = tuple(np.random.randint(0, high=img_dim, size=2))
    new_bad_ind_3d = (slice(None),) + new_bad_ind
    add_bad = [new_bad_ind[::-1]]

    bad_cube = np.zeros((n_im, img_dim, img_dim), dtype=bool)
    for i in range(n_im):
        bad_cube[(i, *np.random.randint(0, high=img_dim, size=2))] = 1

    # Put value out of data bounds (0, 1) to make sure corrected != original
    data[bad_cube] = 1e5
    data[new_bad_ind_3d] = 1e5

    cleaned = clean_data(data, sky=False, apod=False, bad_map=bad_cube, add_bad=add_bad)
    nobpix_list = [
        fix_bad_pixels(img, bad_map=bmap, add_bad=add_bad)
        for img, bmap in zip(data, bad_cube)
    ]

    full_bad_cube = bad_cube.copy()
    full_bad_cube[new_bad_ind_3d] = True

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[full_bad_cube] != data[full_bad_cube])
    assert np.all(cleaned[~full_bad_cube] == data[~full_bad_cube])


@pytest.mark.usefixtures("close_figures")
def test_clean_data_bmap_add_bad_3d():
    # Test combination of 3d bad pixels with add_bad 3d as well
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    # N-dim list or arbitrary (len == 1 here) lists of 2-tuples
    new_bad_inds = [
        tuple(np.random.randint(0, high=img_dim, size=2)) for _ in range(n_im)
    ]
    add_bad = [[ind[::-1]] for ind in new_bad_inds]

    bad_cube = np.zeros((n_im, img_dim, img_dim), dtype=bool)
    for i in range(n_im):
        bad_cube[(i, *np.random.randint(0, high=img_dim, size=2))] = 1
        data[(i, *new_bad_inds[i])] = 1e5

    # Put value out of data bounds (0, 1) to make sure corrected != original
    data[bad_cube] = 1e5

    cleaned = clean_data(data, sky=False, apod=False, bad_map=bad_cube, add_bad=add_bad)
    nobpix_list = [
        fix_bad_pixels(data[i], bad_map=bad_cube[i], add_bad=add_bad[i])
        for i in range(n_im)
    ]

    full_bad_cube = bad_cube.copy()
    for i in range(n_im):
        full_bad_cube[(i, *new_bad_inds[i])] = True

    assert np.all([cleaned[i] == nobpix_list[i] for i in range(data.shape[0])])
    assert np.all(cleaned[full_bad_cube] != data[full_bad_cube])
    assert np.all(cleaned[~full_bad_cube] == data[~full_bad_cube])


def test_3d_bad_pix_abad_3d_shape():
    # Test combination of 3d bad pixels with add_bad 3d as well
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    add_bad = [
        [(1, 2)],
    ] * (n_im + 1)

    with pytest.raises(ValueError, match="3D add_bad should have one list per frame"):
        _get_3d_bad_pixels(None, add_bad, data)


def test_3d_bad_pix_bmap_2d_shape():
    # Assert that bad shape raises error in 2d
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_map = np.zeros((img_dim, img_dim - 1), dtype=bool)

    with pytest.raises(
        ValueError, match="2D bad_map should have the same shape as a frame"
    ):
        _get_3d_bad_pixels(bad_map, None, data)


def test_3d_bad_pix_bmap_3d_shape():
    # Assert that bad shape raises error in 3d
    n_im = 5
    img_dim = 80
    data = np.random.random((n_im, img_dim, img_dim))

    bad_map = np.zeros((n_im, img_dim, img_dim - 1), dtype=bool)

    with pytest.raises(
        ValueError, match="3D bad_map should have the same shape as data cube"
    ):
        _get_3d_bad_pixels(bad_map, None, data)


def test_sky_crop_order():
    # Test that order of cropping and subtraction doesn't affect result if inside cropped image

    img_dim = 80
    img = np.random.random((img_dim, img_dim))
    xmax, ymax = np.random.randint(20, high=img_dim - 20, size=2)
    img[ymax, xmax] = img.max() * 3 + 1  # Add max pixel at pre-determined location

    isz = 2 * np.min([xmax, img.shape[1] - xmax - 1, ymax, img.shape[0] - ymax - 1]) + 1

    dr = 2
    r1 = isz // 2 - dr

    # Correct then crop
    correct = sky_correction(img, r1=r1, dr=dr, center=(xmax, ymax))[0]
    correct_crop = crop_max(correct, isz, filtmed=False)[0]

    # Crop then correct
    cropped, center = crop_max(img, isz, filtmed=False)
    crop_correct = sky_correction(cropped, r1=r1, dr=dr)[0]

    # Make sure find expected center
    assert center == (xmax, ymax)

    # Use 10 * machine precision to make sure passes
    assert np.all(np.abs(correct_crop - crop_correct) < 10 * np.finfo(float).eps)


def test_clean_crop_order():
    # Test that clean_data performs both subtraction and cropping as expected

    img_dim = 80
    n_im = 5
    data = np.random.random((n_im, img_dim, img_dim))
    xmax, ymax = np.random.randint(20, high=img_dim - 20, size=2)
    data[:, ymax, xmax] = data.max() * 3 + 1  # Add max pixel at pre-determined location

    isz = (
        2 * np.min([xmax, data.shape[2] - xmax - 1, ymax, data.shape[1] - ymax - 1]) + 1
    )

    dr = 2
    r1 = isz // 2 - dr

    # Clean cube all at once
    cube = data.copy()
    cube_clean = clean_data(cube, isz=isz, r1=r1, dr=dr, apod=False, f_kernel=None)
    img_cube_clean = cube_clean[0]

    # Correct then crop
    img = data.copy()[0]
    correct = sky_correction(img, r1=r1, dr=dr, center=(xmax, ymax))[0]
    correct[correct < 0] = 0
    img_correct_crop = crop_max(correct, isz, filtmed=False)[0]

    assert np.all(np.abs(img_correct_crop - img_cube_clean) < 10 * np.finfo(float).eps)
