from contextlib import ExitStack
import pytest
import numpy as np
from ctapipe.io import EventSource
from astropy.time import Time
import astropy.units as u

import protozfits
from protozfits.CTA_R1_pb2 import CameraConfiguration, Event
from protozfits.Debug_R1_pb2 import DebugEvent, DebugCameraConfiguration
from protozfits.CoreMessages_pb2 import AnyArray

from ctapipe_io_lst.event_time import time_to_cta_high


ANY_ARRAY_TYPE_TO_NUMPY_TYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.int64,
    8: np.uint64,
    9: np.float32,
    10: np.float64,
}

DTYPE_TO_ANYARRAY_TYPE = {v: k for k, v in ANY_ARRAY_TYPE_TO_NUMPY_TYPE.items()}



@pytest.fixture(scope="session")
def dummy_cta_r1_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dummy_cta_r1")


@pytest.fixture(scope="session")
def dummy_cta_r1(dummy_cta_r1_dir):
    # with protozfits.File("test_data/real/R0/20200218/LST-1.1.Run02006.0004.fits.fz") as f:
    # old_camera_config = f.CameraConfig[0]


    stream_paths = [
        dummy_cta_r1_dir / "LST-1.1.Run10000.0000.fits.fz",
        dummy_cta_r1_dir / "LST-1.2.Run10000.0000.fits.fz",
        dummy_cta_r1_dir / "LST-1.3.Run10000.0000.fits.fz",
        dummy_cta_r1_dir / "LST-1.4.Run10000.0000.fits.fz",
    ]

    run_start = Time("2023-05-16T16:06:31.123")
    camera_config = CameraConfiguration(
        tel_id=1,
        local_run_id=10000,
        config_time_s=run_start.unix,
        camera_config_id=1,
        num_modules=265,
        num_pixels=1855,
        num_channels=2,
        data_model_version="1.0",
        num_samples_nominal=40,
        debug=DebugCameraConfiguration(
            cs_serial="???",
            evb_version="evb-dummy",
            cdhs_version="evb-dummy",
        )
    )

    rng = np.random.default_rng()

    streams = []
    with ExitStack() as stack:
        for stream_path in stream_paths:
            stream = stack.enter_context(protozfits.ProtobufZOFits())
            stream.open(str(stream_path))
            stream.move_to_new_table("CameraConfiguration")
            stream.write_message(camera_config)
            stream.move_to_new_table("Events")
            streams.append(stream)

        event_time = run_start
        event_rate = 8000 / u.s

        for event_count in range(800):
            if event_count % 100 == 98:
                event_type = 0
            elif event_count % 100 == 99:
                event_type = 2
            else:
                event_type = 32

            event_time = event_time + rng.exponential(1 / event_rate.to_value(1 / u.s)) * u.s
            event_time_s, event_time_qns = time_to_cta_high(event_time)

            event = Event(
                event_id=event_count + 1,
                tel_id=1,
                local_run_id=10000,
                event_type=event_type,
                event_time_s=int(event_time_s),
                event_time_qns=int(event_time_qns),
            )

            streams[event_count % len(streams)].write_message(event)

    return stream_paths[0]





def test_is_compatible(dummy_cta_r1):
    from ctapipe_io_lst import LSTEventSource
    assert LSTEventSource.is_compatible(dummy_cta_r1)

def test_no_calibration(dummy_cta_r1):
    with EventSource(dummy_cta_r1, apply_drs4_corrections=False, pointing_information=False) as source:
        n_events = 0
        for e in source:
            n_events += 1
        assert n_events == 800
