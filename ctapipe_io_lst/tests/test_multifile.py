import pytest
from pathlib import Path
import os

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_r0_dir = test_data / 'multifile_test'


def test_multifile_streams():
    from ctapipe_io_lst.multifiles import MultiFiles

    path = test_r0_dir / 'LST-1.1.Run00001.0000.fits.fz'

    with MultiFiles(path) as multi_files:
        assert multi_files.n_open_files == 4

        event_count = 0
        for event in multi_files:
            event_count += 1
            assert event.event_id == event_count

        assert event_count == 40


def test_multifile_all_subruns():
    from ctapipe_io_lst.multifiles import MultiFiles

    path = test_r0_dir / 'LST-1.1.Run00001.0000.fits.fz'

    with MultiFiles(path, all_subruns=True) as multi_files:
        assert multi_files.n_open_files == 4

        event_count = 0
        for event in multi_files:
            event_count += 1
            assert event.event_id == event_count

        assert event_count == 200


def test_multifile_last_subrun():
    from ctapipe_io_lst.multifiles import MultiFiles

    path = test_r0_dir / 'LST-1.1.Run00001.0002.fits.fz'

    with MultiFiles(path, all_subruns=True) as multi_files:
        assert multi_files.n_open_files == 4

        event_count = 80
        for event in multi_files:
            event_count += 1
            assert event.event_id == event_count

        assert event_count == 200


    with MultiFiles(path, all_subruns=True, last_subrun=3) as multi_files:
        assert multi_files.n_open_files == 4

        event_count = 80
        for event in multi_files:
            event_count += 1
            assert event.event_id == event_count

        assert event_count == 160


def test_multifile_single():
    from ctapipe_io_lst.multifiles import MultiFiles

    path = test_r0_dir / 'LST-1.3.Run00001.0002.fits.fz'

    with MultiFiles(path, all_streams=True, all_subruns=True) as multi_files:
        assert multi_files.n_open_files == 1

        event_count = 79
        for event in multi_files:
            event_count += 4
            assert event.event_id == event_count
        assert event_count == 119

    path = test_r0_dir / 'LST-1.1.Run00001.0000.fits.fz'
    with MultiFiles(path, all_streams=False, all_subruns=False) as multi_files:
        assert multi_files.n_open_files == 1

        event_count = -3
        for event in multi_files:
            event_count += 4
            assert event.event_id == event_count
        assert event_count == 37
