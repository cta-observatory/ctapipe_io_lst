import os
from pathlib import Path

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_night_summary = test_data / 'real/monitoring/NightSummary/NightSummary_20200218.txt'


def test_read_night_summary():
    from ctapipe_io_lst.event_time import read_night_summary

    summary = read_night_summary(test_night_summary)
    assert summary['ucts_t0_dragon'].dtype == int
