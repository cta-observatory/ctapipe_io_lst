from pkg_resources import resource_filename
import os
from pathlib import Path

example_file_path = Path(resource_filename(
    'protozfits',
    os.path.join(
        'tests',
        'resources',
        'example_LST_R1_10_evts.fits.fz'
    )
))

# ADC_SAMPLES_SHAPE = (2, 14, 40)


def test_loop_over_events():
    from ctapipe_io_lst import LSTEventSource

    n_events = 10
    source = LSTEventSource(
        input_url=example_file_path,
        max_events=n_events
    )

    for i, event in enumerate(source, start=1):
        assert event.r0.tels_with_data == [0]
        for telid in event.r0.tels_with_data:
            assert event.index.event_id == i
            n_gain = 2
            n_camera_pixels = \
                source.subarray.tels[telid].camera.geometry.n_pixels
            num_samples = event.lst.tel[telid].svc.num_samples
            waveform_shape = (n_gain, n_camera_pixels, num_samples)

            assert event.r0.tel[telid].waveform.shape == waveform_shape

    # make sure max_events works
    assert i == n_events


def test_is_compatible():
    from ctapipe_io_lst import LSTEventSource
    assert LSTEventSource.is_compatible(example_file_path)


def test_factory_for_lst_file():
    from ctapipe.io import event_source

    reader = event_source(example_file_path)

    # import here to see if ctapipe detects plugin
    from ctapipe_io_lst import LSTEventSource

    assert isinstance(reader, LSTEventSource)
    assert reader.input_url == example_file_path
