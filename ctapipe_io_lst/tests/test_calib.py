from pkg_resources import resource_filename
import os
from pathlib import Path
import numpy as np

resource_dir = Path(resource_filename(
    'ctapipe_io_lst',
    os.path.join('tests', 'resources')
))

test_calib_path = resource_dir / 'calibration.Run2462.0000.hdf5'
test_drs4_pedestal_path = resource_dir / 'drs4_pedestal.Run2460.0000.fits.gz'


def test_read_calib_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections

    mon = LSTR0Corrections._read_calibration_file(test_calib_path)
    # only one telescope in that file
    assert mon.tel.keys() == {1, }


def test_read_drs4_pedestal_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections, N_CAPACITORS, N_ROI

    pedestal = LSTR0Corrections._read_drs4_pedestal_file(test_drs4_pedestal_path)

    assert pedestal.shape[-1] == N_CAPACITORS + N_ROI
    # check circular boundary
    assert np.all(pedestal[..., :N_ROI] == pedestal[..., N_CAPACITORS:])

    # check offset is applied
    pedestal_offset = LSTR0Corrections._read_drs4_pedestal_file(
        test_drs4_pedestal_path, offset=100,
    )
    assert np.all((pedestal - pedestal_offset) == 100)
