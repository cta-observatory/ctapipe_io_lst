import pytest
from pathlib import Path
import os

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_r0_dir = test_data / 'real/R0/20200218'


def test_multifile():
    from ctapipe_io_lst.multifiles import MultiFiles

    paths = test_r0_dir.glob('LST-1.*.Run02005.0000_first50.fits.fz')

    with MultiFiles(paths) as multi_files:
        event_count = 0
        for event in multi_files:
            event_count += 1
            assert event.event_id == event_count

        assert event_count == 200

def test_different_runs():
    from ctapipe_io_lst.multifiles import MultiFiles

    paths = [
        test_r0_dir / 'LST-1.1.Run02005.0000_first50.fits.fz',
        test_r0_dir / 'LST-1.2.Run02008.0000_first50.fits.fz',
    ]

    with pytest.raises(IOError):
        MultiFiles(paths)
