from ctapipe_io_lst.constants import HIGH_GAIN
import os
from pathlib import Path
from traitlets.config import Config
import numpy as np
import tables
import pkg_resources

resource_dir = Path(pkg_resources.resource_filename(
    'ctapipe_io_lst', 'tests/resources'
))

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_r0_path = test_data / 'real/R0/20200218/LST-1.1.Run02008.0000_first50.fits.fz'
test_r0_calib_path = test_data / 'real/R0/20200218/LST-1.1.Run02006.0004.fits.fz'
test_calib_path = test_data / 'real/calibration/20200218/v05/calibration.Run2006.0000.hdf5'
test_drs4_pedestal_path = test_data / 'real/calibration/20200218/v05/drs4_pedestal.Run2005.0000.fits'
test_time_calib_path = test_data / 'real/calibration/20200218/v05/time_calibration.Run2006.0000.hdf5'
test_missing_module_path = test_data / 'real/R0/20210215/LST-1.1.Run03669.0000_first50.fits.fz'




def test_get_first_capacitor():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.calibration import (
        get_first_capacitors_for_pixels,
        N_GAINS, N_PIXELS_MODULE, N_MODULES,
    )

    tel_id = 1
    source = LSTEventSource(test_r0_calib_path)
    event = next(iter(source))

    first_capacitor_id = event.lst.tel[tel_id].evt.first_capacitor_id

    with tables.open_file(resource_dir / 'first_caps.hdf5', 'r') as f:
        expected = f.root.first_capacitor_for_modules[:]

    first_caps = get_first_capacitors_for_pixels(first_capacitor_id)

    # we used a different shape (N_MODULES, N_GAINS, N_PIXELS_MODULE) before
    # so have to reshape to be able to compare
    first_caps = first_caps.reshape((N_GAINS, N_MODULES, N_PIXELS_MODULE))
    first_caps = np.swapaxes(first_caps, 0, 1)
    assert np.all(first_caps == expected)


def test_read_calib_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections

    mon = LSTR0Corrections._read_calibration_file(test_calib_path)
    # only one telescope in that file
    assert mon.tel.keys() == {1, }


def test_read_drs4_pedestal_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections, N_CAPACITORS_PIXEL, N_SAMPLES

    pedestal = LSTR0Corrections._get_drs4_pedestal_data(test_drs4_pedestal_path)

    assert pedestal.shape[-1] == N_CAPACITORS_PIXEL + N_SAMPLES
    # check circular boundary
    assert np.all(pedestal[..., :N_SAMPLES] == pedestal[..., N_CAPACITORS_PIXEL:])

    # check offset is applied
    pedestal_offset = LSTR0Corrections._get_drs4_pedestal_data(
        test_drs4_pedestal_path, offset=100,
    )
    assert np.all((pedestal - pedestal_offset) == 100)


def test_read_drs_time_calibration_file():
    from ctapipe_io_lst.calibration import LSTR0Corrections, N_GAINS, N_PIXELS

    fan, fbn = LSTR0Corrections.load_drs4_time_calibration_file(test_time_calib_path)

    assert fan.shape == fbn.shape
    assert fan.shape[0] == N_GAINS
    assert fan.shape[1] == N_PIXELS


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


def test_source_with_all():
    from ctapipe_io_lst import LSTEventSource

    config = Config({
        'LSTEventSource': {
            'LSTR0Corrections': {
                'drs4_pedestal_path': test_drs4_pedestal_path,
                'drs4_time_calibration_path': test_time_calib_path,
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
            assert np.any(event.calibration.tel[1].dl1.time_shift != 0)


def test_missing_module():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.constants import N_PIXELS_MODULE, N_SAMPLES

    config = Config({
        'LSTEventSource': {
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
            assert np.all(waveform[failing_pixels[HIGH_GAIN]] == 0)

def test_no_gain_selection():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.constants import N_PIXELS, N_GAINS, N_SAMPLES

    config = Config({
        'LSTEventSource': {
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


def test_timelapse_no_underflow():
    '''Test that the timelapse calibration does not create an underflow'''
    from ctapipe_io_lst.calibration import do_time_lapse_corr
    from ctapipe_io_lst.constants import (
        N_GAINS, N_PIXELS, N_SAMPLES, N_MODULES,
        N_CAPACITORS_PIXEL, CLOCK_FREQUENCY_KHZ,
    )
    from ctapipe_io_lst.calibration import ped_time

    # low value that should underflow if not checked
    waveform = np.full((N_GAINS, N_PIXELS, N_SAMPLES), 5, dtype=np.uint16)

    # 0.1 ms should result in a correction around 50 ADC samples, well above 5
    delta_t_ms = 0.1
    local_clock_counter = np.full(N_MODULES, delta_t_ms * CLOCK_FREQUENCY_KHZ, dtype=np.uint64)

    # make sure we would actually trigger an underflow
    assert np.all(ped_time(delta_t_ms) > waveform)

    first_capacitors = np.zeros((N_GAINS, N_PIXELS), dtype=np.int16)
    last_readout_time = np.ones((N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL), dtype=np.uint64)
    expected_pixels_id = np.arange(N_PIXELS)

    do_time_lapse_corr(
        waveform,
        local_clock_counter,
        first_capacitors,
        last_readout_time,
        expected_pixels_id,
    )
    assert np.all(waveform == 0)

def test_no_gain_selection_no_drs4time_calib():
    from ctapipe_io_lst import LSTEventSource
    from ctapipe_io_lst.constants import N_PIXELS, N_GAINS, N_SAMPLES

    config = Config({
        'LSTEventSource': {
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
