from ctapipe.io import EventSource

path = "../data/cta_r1_lst_dummy/Unknown_20230403_0000.fits.fz"


def test_is_compatible():
    from ctapipe_io_lst import LSTEventSource
    assert LSTEventSource.is_compatible(path)

def test_no_calibration():
    with EventSource(path, apply_drs4_corrections=False, pointing_information=False) as source:
        n_events = 0
        for e in source:
            n_events += 1
        assert n_events > 0
