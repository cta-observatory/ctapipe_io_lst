import os
from pathlib import Path

import numpy as np
import protozfits
import pytest

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_file_protor1 = test_data / "real/R0/20200218/LST-1.1.Run02008.0000_first50.fits.fz"
test_file_r1v1 = test_data / "real/R0/20231219/LST-1.1.Run16255.0000_first50.fits.fz"


@pytest.mark.parametrize("path", (test_file_protor1, test_file_r1v1))
def test_get_pixel_table_from_camera_config(path):
    from ctapipe_io_lst.pixels import get_pixel_table_from_camera_config

    with protozfits.File(str(path)) as f:
        config = getattr(f, "CameraConfig", getattr(f, "CameraConfiguration", None))[0]

    table = get_pixel_table_from_camera_config(config)

    np.testing.assert_array_equal(np.unique(table["module_id"]), np.arange(265))

    # check relationship between hardware_pixel_id and position in module
    np.testing.assert_array_equal(
        table["module_id"] * 7 + table["module_pixel_index"],
        table["hardware_pixel_id"],
    )

    # check we have sorted by pixel id
    np.testing.assert_array_equal(table["pixel_id"], np.arange(1855))
