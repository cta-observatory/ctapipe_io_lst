import os
from importlib.resources import files, as_file
from pathlib import Path
import pickle

import numpy as np
import pytest
import tables
from traitlets.config import Config

from ctapipe_io_lst.constants import HIGH_GAIN
from ctapipe_io_lst.compat import CTAPIPE_GE_0_21


resource_dir = files('ctapipe_io_lst') / 'tests/resources'
test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_r0_path = test_data / 'real/R0/20200218/LST-1.1.Run02008.0000_first50.fits.fz'
test_r0_calib_path = test_data / 'real/R0/20200218/LST-1.1.Run02006.0004.fits.fz'
test_missing_module_path = test_data / 'real/R0/20210215/LST-1.1.Run03669.0000_first50.fits.fz'
test_r0_gainselected_path = test_data / 'real/R0/20200218/LST-1.1.Run02008.0000_first50_gainselected.fits.fz'

calib_version = "ctapipe-v0.17"
calib_path = test_data / 'real/monitoring/PixelCalibration/Cat-A/'
test_calib_path = calib_path / f'calibration/20200218/{calib_version}/calibration_filters_52.Run02006.0000.h5'
test_calib_path_fits = calib_path / f'calibration/20200218/{calib_version}/calibration_filters_52.Run02006.0000.fits.gz'
test_drs4_pedestal_path = calib_path / f'drs4_baseline/20200218/{calib_version}/drs4_pedestal.Run02005.0000.h5'
test_drs4_pedestal_path_fits = calib_path / f'drs4_baseline/20200218/{calib_version}/drs4_pedestal.Run02005.0000.fits.gz'
test_time_calib_path = calib_path / f'drs4_time_sampling_from_FF/20191124/{calib_version}/time_calibration.Run01625.0000.h5'
test_time_calib_path_fits = calib_path / f'drs4_time_sampling_from_FF/20191124/{calib_version}/time_calibration.Run01625.0000.fits.gz'


@pytest.mark.parametrize(
    "array",
    [
        np.array([1, 2], dtype=">i4"),
        np.array([1, 2], dtype="<i4"),
        np.array([1, 2], dtype="i4"),
    ]
)
def test_to_native(array):
    from ctapipe_io_lst.calibration import to_native

    converted = to_native(array)
    assert converted.dtype.isnative
    np.testing.assert_array_equal(converted, array)

def test_get_first_capacitor():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.calibration import (
        get_first_capacitors_for_pixels,
        N_GAINS, N_PIXELS_MODULE, N_MODULES,
    )

    tel_id = 1
    source = LSTEventSource(
        test_r0_calib_path,
        apply_drs4_corrections=False,
        pointing_information=False,
    )
    event = next(iter(source))

    first_capacitor_id = event.lst.tel[tel_id].evt.first_capacitor_id

    with as_file(resource_dir / 'first_caps.hdf5') as path:
        with tables.open_file(path, 'r') as f:
            expected = f.root.first_capacitor_for_modules[:]

    first_caps = get_first_capacitors_for_pixels(first_capacitor_id)

    # we used a different shape (N_MODULES, N_GAINS, N_PIXELS_MODULE) before
    # so have to reshape to be able to compare
    first_caps = first_caps.reshape((N_GAINS, N_MODULES, N_PIXELS_MODULE))
    first_caps = np.swapaxes(first_caps, 0, 1)
    assert np.all(first_caps == expected)



@pytest.mark.parametrize(
    "path",
    [
        pytest.param(test_calib_path, id="hdf5"),
        pytest.param(test_calib_path_fits, id="fits"),
    ]
)
def test_read_calib_file(path):
    from ctapipe_io_lst.calibration import LSTR0Corrections

    mon = LSTR0Corrections._read_calibration_file(path)
    # only one telescope in that file
    assert mon.tel.keys() == {1, }


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(test_drs4_pedestal_path, id="hdf5"),
        pytest.param(test_drs4_pedestal_path_fits, id="fits"),
    ]
)
def test_read_drs4_pedestal_file(path):
    from ctapipe_io_lst.calibration import LSTR0Corrections, N_CAPACITORS_PIXEL, N_SAMPLES

    pedestal = LSTR0Corrections._get_drs4_pedestal_data(path, tel_id=1)

    assert pedestal.shape[-1] == N_CAPACITORS_PIXEL + N_SAMPLES
    # check circular boundary
    assert np.all(pedestal[..., :N_SAMPLES] == pedestal[..., N_CAPACITORS_PIXEL:])

@pytest.mark.parametrize(
    "path",
    [
        pytest.param(test_time_calib_path, id="hdf5"),
        pytest.param(test_time_calib_path_fits, id="fits"),
    ]
)

def test_read_drs_time_calibration_file(path):
    from ctapipe_io_lst.calibration import LSTR0Corrections, N_GAINS, N_PIXELS

    fan, fbn = LSTR0Corrections.load_drs4_time_calibration_file(path)

    assert fan.shape == fbn.shape
    assert fan.shape[0] == N_GAINS
    assert fan.shape[1] == N_PIXELS


def test_init():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.calibration import LSTR0Corrections

    subarray = LSTEventSource.create_subarray()
    r0corr = LSTR0Corrections(subarray)
    assert r0corr.last_readout_time.keys() == {1, }


def test_source_with_drs4_pedestal():
    from ctapipe_io_lst import LSTEventSource

    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
            },
        },
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
            'pointing_information': False,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'calibration_path': test_calib_path,
            },
        },
    })

    source = LSTEventSource(
        input_url=test_r0_path,
        config=config,
    )

    assert source.r0_r1_calibrator.mon_data is not None
    with source:
        for event in source:
            assert event.r1.tel[1].waveform is not None

fits_calibration = {
    "drs4_pedestal_path": test_drs4_pedestal_path,
    "drs4_time_calibration_path": test_time_calib_path,
    "calibration_path": test_calib_path,
}
hdf5_calibration = {
    "drs4_pedestal_path": test_drs4_pedestal_path_fits,
    "drs4_time_calibration_path": test_time_calib_path_fits,
    "calibration_path": test_calib_path_fits,
}
@pytest.mark.parametrize("trigger_information", [True, False])
@pytest.mark.parametrize(
    "calib_config",
    (
        pytest.param(hdf5_calibration, id="hdf5"),
        pytest.param(fits_calibration, id="fits"),
    )
)

def test_calibration(trigger_information,calib_config):
    from ctapipe_io_lst import LSTEventSource

    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'trigger_information': trigger_information, 
            'LSTR0Corrections': calib_config
        },
    })

    source = LSTEventSource(
        input_url=test_r0_path,
        config=config,
    )

    assert source.r0_r1_calibrator.mon_data is not None
    with source:
        for event in source:
            assert event.r1.tel[1].waveform is not None
            assert np.any(event.calibration.tel[1].dl1.time_shift != 0)

def test_missing_module():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.constants import N_PIXELS_MODULE, N_SAMPLES

    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'drs4_time_calibration_path': test_time_calib_path,
                'calibration_path': test_calib_path,
            },
        },
    })

    source = LSTEventSource(
        input_url=test_missing_module_path,
        config=config,
    )

    assert source.r0_r1_calibrator.mon_data is not None
    with source:
        for event in source:
            waveform = event.r1.tel[1].waveform
            assert waveform is not None


            failing_pixels = event.mon.tel[1].pixel_status.hardware_failing_pixels

            # one module failed, in each gain channel
            assert np.count_nonzero(failing_pixels) ==  2 * N_PIXELS_MODULE

            # there might be zeros in other pixels than just the broken ones
            assert np.count_nonzero(waveform == 0) >= N_PIXELS_MODULE * (N_SAMPLES - 4)

            # waveforms in failing pixels must be all 0
            if CTAPIPE_GE_0_21:
                np.testing.assert_equal(waveform[:, failing_pixels[HIGH_GAIN]], 0)
            else:
                np.testing.assert_equal(waveform[failing_pixels[HIGH_GAIN]], 0)

def test_no_gain_selection():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.constants import N_PIXELS, N_GAINS, N_SAMPLES

    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'drs4_time_calibration_path': test_time_calib_path,
                'calibration_path': test_calib_path,
                'select_gain': False,
            },
        },
    })

    source = LSTEventSource(
        input_url=test_r0_calib_path,
        config=config,
    )

    assert source.r0_r1_calibrator.mon_data is not None
    with source:
        for event in source:
            assert event.r1.tel[1].waveform is not None
            assert event.r1.tel[1].waveform.ndim == 3
            assert event.r1.tel[1].waveform.shape == (N_GAINS, N_PIXELS, N_SAMPLES - 4)


def test_no_gain_selection_no_drs4time_calib():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.constants import N_PIXELS, N_GAINS, N_SAMPLES

    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'calibration_path': test_calib_path,
                'select_gain': False,
            },
        },
    })

    source = LSTEventSource(
        input_url=test_r0_calib_path,
        config=config,
    )

    assert source.r0_r1_calibrator.mon_data is not None
    with source:
        for event in source:
            assert event.r1.tel[1].waveform is not None
            assert event.r1.tel[1].waveform.ndim == 3
            assert event.r1.tel[1].waveform.shape == (N_GAINS, N_PIXELS, N_SAMPLES - 4)


def test_already_gain_selected():
    from ctapipe_io_lst import LSTEventSource

    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'drs4_time_calibration_path': test_time_calib_path,
                'calibration_path': test_calib_path,
            },
        },
    })

    source = LSTEventSource(test_r0_gainselected_path, config=config)
    reference_source = LSTEventSource(test_r0_path, config=config)

    with source, reference_source:
        for event, reference_event in zip(source, reference_source):
            assert np.all(event.r1.tel[1].waveform == reference_event.r1.tel[1].waveform)
    assert event.count == 199


def test_spike_positions():
    from ctapipe_io_lst.calibration import get_spike_A_positions
    from ctapipe_io_lst.constants import N_CAPACITORS_PIXEL

    positions = {}
    for current in range(N_CAPACITORS_PIXEL):
        for previous in range(N_CAPACITORS_PIXEL):
            pos = get_spike_A_positions(current, previous)
            if pos:
                positions[(current, previous)] = pos

    with (test_data / 'spike_positions.pickle').open('rb') as f:
        expected_positions = pickle.load(f)

    for key, pos in positions.items():
        assert sorted(pos) == sorted(expected_positions[key])


def test_calibrate_precalibrated():
    from ctapipe_io_lst import LSTEventSource

    # file with pe calibration applied by EVB
    test_file = "20231219/LST-1.1.Run16255.0000_first50.fits.fz"

    path = test_data / "real/R0" / test_file
    config = Config({
        'LSTEventSource': {
            'pointing_information': False,
            'apply_drs4_corrections': True,
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'drs4_time_calibration_path': test_time_calib_path,
                'calibration_path': test_calib_path,
            },
        },
    })


    previous = None
    with LSTEventSource(path, config=config) as source:
        n_read = 0
        for event in source:
            n_read += 1

            time_shift = event.calibration.tel[1].dl1.time_shift
            # test we filled the timeshift although the data is pre-calibrated
            assert time_shift is not None
            assert np.any(time_shift != 0)

            # check that time_shift varies from event to event due to drs4 time shift
            # regression test for the bug of not updating first_capacitors
            if previous is not None and time_shift.shape == previous.shape:
                assert np.any(time_shift != previous)
            previous = time_shift

        assert n_read == 200
