import os
from pathlib import Path
from traitlets.config import Config
import numpy as np
import tables

resource_dir = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))

test_r0_path = resource_dir / 'real/R0/20200218/LST-1.1.Run02006.0004.fits.fz'
test_calib_path = resource_dir / 'real/calibration/20200218/v05/calibration.Run2006.0000.hdf5'
test_drs4_pedestal_path = resource_dir / 'real/calibration/20200218/v05/drs4_pedestal.Run2005.0000.fits'
test_time_calib_path = resource_dir / 'real/calibration/20200218/v05/time_calibration.Run2006.0000.hdf5'


def test_get_first_capacitor():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.calibration import get_first_capacitor_for_modules

    tel_id = 1
    source = LSTEventSource(test_r0_path)
    event = next(iter(source))

    first_capacitor_id = event.lst.tel[tel_id].evt.first_capacitor_id

    with tables.open_file('./first_caps.hdf5', 'r') as f:
        expected = f.root.first_capacitor_for_modules[:]

    first_caps = get_first_capacitor_for_modules(first_capacitor_id)
    assert np.all(first_caps == expected)


def test_read_calib_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections

    mon = LSTR0Corrections._read_calibration_file(test_calib_path)
    # only one telescope in that file
    assert mon.tel.keys() == {1, }


def test_read_drs4_pedestal_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections, N_CAPACITORS_4, N_ROI

    pedestal = LSTR0Corrections._get_drs4_pedestal_data(test_drs4_pedestal_path)

    assert pedestal.shape[-1] == N_CAPACITORS_4 + N_ROI
    # check circular boundary
    assert np.all(pedestal[..., :N_ROI] == pedestal[..., N_CAPACITORS_4:])

    # check offset is applied
    pedestal_offset = LSTR0Corrections._get_drs4_pedestal_data(
        test_drs4_pedestal_path, offset=100,
    )
    assert np.all((pedestal - pedestal_offset) == 100)


def test_init():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.calibration import LSTR0Corrections

    subarray = LSTEventSource.create_subarray(geometry_version=4)
    r0corr = LSTR0Corrections(subarray)
    assert r0corr.last_readout_time.keys() == {1, }


def test_source_with_drs4_pedestal():
    from ctapipe_io_lst import LSTEventSource

    config = Config({
        'LSTEventSource': {
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
            }
        }
    })

    source = LSTEventSource(
        input_url=test_r0_path,
        config=config,
    )
    assert source.r0_r1_calibrator.drs4_pedestal_path.tel[1] == test_drs4_pedestal_path.absolute()

    with source:
        for event in source:
            assert event.r1.tel[1].waveform is not None


def test_source_with_calibration():
    from ctapipe_io_lst import LSTEventSource

    config = Config({
        'LSTEventSource': {
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'calibration_path': test_calib_path,
            }
        }
    })

    source = LSTEventSource(
        input_url=test_r0_path,
        config=config,
    )

    assert source.r0_r1_calibrator.mon_data is not None
    with source:
        for event in source:
            assert event.r1.tel[1].waveform is not None
