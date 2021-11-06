import numpy as np
from amical.data_processing import sky_correction
import pytest


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
