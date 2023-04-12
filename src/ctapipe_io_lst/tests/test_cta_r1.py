path = "../data/cta_r1_lst_dummy/Unknown_20230403_0000.fits.fz"


def test_is_compatible():
    from ctapipe_io_lst import LSTEventSource

    assert LSTEventSource.is_compatible(path)
