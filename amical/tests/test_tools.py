import numpy as np
import pytest

from amical import tools


def test_crop_max():

    img_size = 80  # Same size as NIRISS images
    img = np.ones((img_size, img_size))
    xmax, ymax = 17, 57
    img[ymax, xmax] = img.max() * 5  # Add off-centered max pixel

    # Pre-calculate expected max size
    isz_max = 2 * np.min([xmax, img.shape[1] - xmax, ymax, img.shape[0] - ymax]) + 1
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
