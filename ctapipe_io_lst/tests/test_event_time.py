import logging
import os
from pathlib import Path
import numpy as np

from astropy.table import Table
from astropy.time import Time
import astropy.units as u

from ctapipe_io_lst.constants import N_MODULES
from ctapipe_io_lst.containers import LSTArrayEventContainer

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_night_summary = test_data / 'real/monitoring/NightSummary/NightSummary_20200218.txt'
test_night_summary_with_nans = test_data / 'real/monitoring/NightSummary/NightSummary_20201215.txt'


int_cols = [
    'first_valid_event_dragon',
    'ucts_t0_dragon',
    'dragon_counter0',
    'first_valid_event_tib',
    'ucts_t0_tib',
    'tib_counter0',
]


def test_time_unix_tai():
    t = Time('2020-01-01T00:00:00', scale='utc')
    assert (t.unix_tai - t.unix) == 37


def test_read_night_summary():
    from ctapipe_io_lst.event_time import read_night_summary

    summary = read_night_summary(test_night_summary)

    for col in int_cols:
        assert summary[col].dtype == np.int64


def test_read_night_summary_missing():
    from ctapipe_io_lst.event_time import read_night_summary

    summary = read_night_summary(test_night_summary_with_nans)
    for col in int_cols:
        assert summary[col].dtype == np.int64

    assert len(summary) == 26
    assert isinstance(summary['timestamp'], Time)
    assert np.ma.is_masked(summary['tib_counter0'])
    assert np.count_nonzero(summary['tib_counter0'].mask) == 12


def test_ucts_jumps():
    '''
    We creat toy data that will have two ucts jumps.
    When a ucts event goes missing, the event actually contains
    the ucts data for the next event. For a second jump, the events are
    no out of sync by two.
    We are going to just use event ids for the timestamps
    '''
    from ctapipe_io_lst.event_time import EventTimeCalculator
    from ctapipe_io_lst import LSTEventSource
    tel_id = 1

    event = LSTArrayEventContainer()
    lst = event.lst.tel[tel_id]
    lst.evt.extdevices_presence = 0b1111_1111
    lst.svc.module_ids = np.arange(N_MODULES)

    subarray = LSTEventSource.create_subarray(geometry_version=4, tel_id=1)

    true_time_s = int(Time.now().unix)

    s_to_ns = int(1e9)
    time_calculator = EventTimeCalculator(
        subarray=subarray,
        dragon_reference_time=true_time_s * s_to_ns,
        dragon_reference_counter=0,
        timestamp='ucts'  # use ucts to make sure we identify jumps and fallback to tib
    )

    n_events = 22
    true_event_id = np.arange(n_events)
    # One event every 10 us.
    true_time_ns = 10 * true_event_id * int(1e3)

    # no jumps
    table = Table({
        'event_id': true_event_id,
        'ucts_timestamp': true_time_s * s_to_ns + true_time_ns,
        'tib_pps_counter': np.full(n_events, 0),
        'tib_tenMHz_counter': (true_time_ns / 100).astype(int),
        'pps_counter': [np.full(N_MODULES, 0) for _ in range(n_events)],
        'tenMHz_counter': [np.full(N_MODULES, int(t / 100)) for t in true_time_ns],
        # to check if we handle jumps correctly, we put the event id here
        'ucts_trigger_type': np.arange(n_events),
        'tib_masked_trigger': np.arange(n_events),
    })

    for i in range(n_events):
        for col in table.colnames:
            setattr(lst.evt, col, table[col][i])

        time_calculator(tel_id, event)
        assert len(time_calculator.previous_ucts_timestamps[tel_id]) == 0


    # no we introduce three jumps
    for col in ('ucts_timestamp', 'ucts_trigger_type'):
        table[col][5:-1] = table[col][6:]
        table[col][12:-1] = table[col][13:]
        table[col][15:-1] = table[col][16:]
    table = table[:-3]


    ucts_trigger_types = []
    last_time = None
    for i in range(n_events - 3):
        for col in table.colnames:
            setattr(lst.evt, col, table[col][i])
        event.index.event_id = table['event_id'][i]

        time = time_calculator(tel_id, event)

        if last_time is not None:
            # timestamp is only accurate to 1 us due to floating point
            assert np.isclose((time - last_time).to_value(u.us), 10, atol=0.5)
        last_time = time

        if i < 5:
            assert len(time_calculator.previous_ucts_timestamps[tel_id]) == 0
        elif i < 13:
            assert len(time_calculator.previous_ucts_timestamps[tel_id]) == 1
        elif i < 17:
            assert len(time_calculator.previous_ucts_timestamps[tel_id]) == 2

        ucts_trigger_types.append(lst.evt.ucts_trigger_type)

    assert np.all(np.array(ucts_trigger_types) == table['event_id'])



def test_no_reference(caplog):
    '''
    Test that we switch to `event.lst.svc.date` as reference when
    no reference is available.
    '''
    from ctapipe_io_lst.event_time import EventTimeCalculator
    from ctapipe_io_lst import LSTEventSource

    caplog.set_level(logging.WARNING)

    tel_id = 1
    event = LSTArrayEventContainer()
    lst = event.lst.tel[tel_id]
    lst.evt.extdevices_presence = 0b1111_1101
    lst.svc.module_ids = np.arange(N_MODULES)

    subarray = LSTEventSource.create_subarray(geometry_version=4, tel_id=1)

    first_event_time = Time.now()
    # run start is a couple of seconds earlier than first event
    lst.svc.date = (first_event_time - 5 * u.s).unix

    s_to_ns = int(1e9)

    n_events = 5
    true_event_id = np.arange(n_events)
    true_time_s = int(first_event_time.unix)
    # one event every 100 micro seconds
    true_time_ns = int(1e5) * true_event_id

    time_calculator = EventTimeCalculator(
        subarray=subarray,
    )

    table = Table({
        'event_id': true_event_id,
        'ucts_timestamp': np.zeros(n_events, dtype=int),
        'pps_counter': [np.full(N_MODULES, 100) for _ in range(n_events)],
        'tenMHz_counter': [np.full(N_MODULES, int(t / 100)) for t in true_time_ns],
    })

    for i in range(n_events):
        for col in table.colnames:
            setattr(lst.evt, col, table[col][i])

        time = time_calculator(tel_id, event)
        expected_time = first_event_time + (i * 100 * u.us - 5 * u.s)
        # precision of float64 timestamp around 300 ns
        assert np.abs((time - expected_time).to_value(u.us)) < 0.5

    warning_found = False
    for record in caplog.records:
        if 'Cannot calculate a precise timestamp' in record.message:
            warning_found = True
    assert warning_found
