import numpy as np
import os
from pathlib import Path
import tempfile
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS_MODULE, N_SAMPLES
from traitlets.config import Config

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_r0_dir = test_data / 'real/R0/20200218'
test_r0_path = test_r0_dir / 'LST-1.1.Run02006.0004.fits.fz'
test_r0_path_all_streams = test_r0_dir / 'LST-1.1.Run02008.0000_first50.fits.fz'

test_missing_module_path = test_data / 'real/R0/20210215/LST-1.1.Run03669.0000_first50.fits.fz'

# ADC_SAMPLES_SHAPE = (2, 14, 40)


config = Config()
config.LSTEventSouce.EventTimeCalculator.extract_reference = True


def test_loop_over_events():
    from ctapipe_io_lst import LSTEventSource

    n_events = 10
    source = LSTEventSource(
        input_url=test_r0_path,
        max_events=n_events,
    )

    for i, event in enumerate(source):
        assert event.count == i
        for telid in event.r0.tel.keys():
            n_gains = 2
            n_pixels = source.subarray.tels[telid].camera.geometry.n_pixels
            n_samples = event.lst.tel[telid].svc.num_samples
            waveform_shape = (n_gains, n_pixels, n_samples)
            assert event.r0.tel[telid].waveform.shape == waveform_shape
            assert event.mon.tel[telid].pixel_status.hardware_failing_pixels.shape == (n_gains, n_pixels)

    # make sure max_events works
    assert (i + 1) == n_events


def test_multifile():
    from ctapipe_io_lst import LSTEventSource

    source = LSTEventSource(input_url=test_r0_path_all_streams)
    assert len(set(source.file_list)) == 4

    for i, event in enumerate(source):
        # make sure all events are present and in the correct order
        assert event.index.event_id == i + 1

    # make sure we get all events from all streams (50 per stream)
    assert (i + 1) == 200


def test_is_compatible():
    from ctapipe_io_lst import LSTEventSource
    assert LSTEventSource.is_compatible(test_r0_path)


def test_event_source_for_lst_file():
    from ctapipe.io import EventSource

    reader = EventSource(test_r0_path)

    # import here to see if ctapipe detects plugin
    from ctapipe_io_lst import LSTEventSource

    assert isinstance(reader, LSTEventSource)
    assert reader.input_url == test_r0_path


def test_subarray():
    from ctapipe_io_lst import LSTEventSource

    source = LSTEventSource(test_r0_path)
    subarray = source.subarray
    subarray.info()
    subarray.to_table()

    assert source.lst_service.telescope_id == 1
    assert source.lst_service.num_modules == 265

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        subarray.to_hdf(f.name)


def test_missing_modules():
    from ctapipe_io_lst import LSTEventSource
    source = LSTEventSource(test_missing_module_path)

    assert source.lst_service.telescope_id == 1
    assert source.lst_service.num_modules == 264

    fill = np.iinfo(np.uint16).max
    for event in source:
        # one module missing, so 7 pixels
        assert np.count_nonzero(event.mon.tel[1].pixel_status.hardware_failing_pixels) == N_PIXELS_MODULE * N_GAINS
        assert np.count_nonzero(event.r0.tel[1].waveform == fill) == N_PIXELS_MODULE * N_SAMPLES * N_GAINS

        # 514 is one of the missing pixels
        assert np.all(event.r0.tel[1].waveform[:, 514] == fill)
