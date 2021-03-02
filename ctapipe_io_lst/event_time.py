from collections import deque, defaultdict
import numpy as np

import astropy.version
from astropy.io.ascii import convert_numpy
from astropy.table import Table
from astropy.time import Time, TimeUnixTai, TimeFromEpoch

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import IntTelescopeParameter, TelescopeParameter

from traitlets import Enum, Int as _Int, Bool


if astropy.version.major == 4 and astropy.version.minor <= 2 and astropy.version.bugfix <= 0:
    # clear the cache to not depend on import orders
    TimeFromEpoch.__dict__['_epoch']._cache.clear()
    # fix for astropy #11245, epoch was wrong by 8 seconds
    TimeUnixTai.epoch_val = '1970-01-01 00:00:00.0'
    TimeUnixTai.epoch_scale = 'tai'



CENTRAL_MODULE = 132


# fix for https://github.com/ipython/traitlets/issues/637
class Int(_Int):
    def validate(self, obj, value):
        if value is None and self.allow_none is True:
            return value

        return super().validate(obj, value)


def calc_dragon_time(lst_event_container, module_index, reference_time, reference_counter):
    '''
    Calculate a unix tai timestamp (in ns) from dragon counter values
    and reference time / counter value for a given module index.
    '''
    pps_counter = lst_event_container.evt.pps_counter[module_index]
    tenMHz_counter = lst_event_container.evt.tenMHz_counter[module_index]
    return (
        reference_time
        + combine_counters(pps_counter, tenMHz_counter)
        - reference_counter
    )


def combine_counters(pps_counter, tenMHz_counter):
    '''
    Combines the values of pps counter and tenMHz_counter
    and returns the sum in ns.
    '''
    return int(1e9) * pps_counter + 100 * tenMHz_counter


def datetime_cols_to_time(date, time):
    return Time(np.char.add(
        date,
        np.char.add('T', time)
    ))


def read_night_summary(path):
    '''
    Read a night summary file into an astropy table

    Parameters
    ----------
    path: str or Path
        Path to the night summary file

    Returns
    -------
    table: Table
        astropy table of the night summary file.
        The columns will have the correct dtype (int64) for the counters
        and missing values (nan in the file) are masked.
    '''

    # convertes for each column to make sure we use the correct
    # dtypes. The counter values in ns are so large that they cannot
    # be exactly represented by float64 values, we need int64
    converters = {
        'run': [convert_numpy(np.int32)],
        'n_subruns': [convert_numpy(np.int32)],
        'run_type': [convert_numpy(str)],
        'date': [convert_numpy(str)],
        'time': [convert_numpy(str)],
        'first_valid_event_dragon': [convert_numpy(np.int64)],
        'ucts_t0_dragon': [convert_numpy(np.int64)],
        'dragon_counter0': [convert_numpy(np.int64)],
        'first_valid_event_tib': [convert_numpy(np.int64)],
        'ucts_t0_tib': [convert_numpy(np.int64)],
        'tib_counter0': [convert_numpy(np.int64)],
    }

    summary = Table.read(
        str(path),
        format='ascii.basic',
        delimiter=' ',
        header_start=0,
        data_start=0,
        names=[
            'run', 'n_subruns', 'run_type', 'date', 'time',
            'first_valid_event_dragon', 'ucts_t0_dragon', 'dragon_counter0',
            'first_valid_event_tib', 'ucts_t0_tib', 'tib_counter0',
        ],
        converters=converters,
        fill_values=("nan", -1),
        guess=False,
        fast_reader=False,
    )

    summary.add_index(['run'])
    summary['timestamp'] = datetime_cols_to_time(summary['date'], summary['time'])
    return summary



def time_from_unix_tai_ns(unix_tai_ns):
    '''
    Create an astropy Time instance from a unix time tai timestamp in ns.
    By using both arguments to time, the result will be a higher precision
    timestamp.
    '''
    full_seconds = unix_tai_ns // int(1e9)
    fractional_seconds = (unix_tai_ns % int(1e9)) * 1e-9
    return Time(full_seconds, fractional_seconds, format='unix_tai')



class EventTimeCalculator(TelescopeComponent):
    '''
    Class to calculate event times from low-level counter information.

    Also keeps track of "UCTS jumps", where UCTS info goes missing for
    a certain event and all following info has to be shifted.


    There are several sources of timing information in LST raw data.

    Each dragon module has two high precision counters, which however only
    give a relative time.
    Same is true for the TIB.

    The only precise absolute timestamp is the UCTS timestamp.
    However, at least during the commissioning, UCTS was/is not reliable
    enough to only use the UCTS timestamp.

    Instead, we calculate an absolute timestamp by using one valid pair
    of dragon counter / ucts timestamp and then use the relative time elapsed
    from this reference using the dragon counter.

    For runs where no such UCTS reference exists, for example because UCTS
    was completely unavailable, we use the start of run timestamp from the
    camera configuration.
    This will however result in imprecises timestamps off by several seconds.
    These might be good enough for interpolating pointing information but
    are only precise for relative time changes, i.e. not suitable for pulsar
    analysis or matching events with MAGIC.
    '''

    timestamp = TelescopeParameter(
        trait=Enum(['ucts', 'dragon']), default_value='dragon'
    ).tag(config=True)

    dragon_reference_time = TelescopeParameter(
        Int(allow_none=True),
        default_value=None,
        help='Reference timestamp for the dragon time calculation in ns'
    ).tag(config=True)

    dragon_reference_counter = TelescopeParameter(
        Int(allow_none=True),
        help='Dragon board counter value of a valid ucts/dragon counter combination',
        default_value=None,
    ).tag(config=True)

    dragon_module_id = IntTelescopeParameter(
        default_value=CENTRAL_MODULE,
        help='Module id used to calculate dragon time.',
    ).tag(config=True)


    def __init__(self, subarray, config=None, parent=None, **kwargs):
        '''Initialize EventTimeCalculator'''
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self.previous_ucts_timestamps = defaultdict(deque)
        self.previous_ucts_trigger_types = defaultdict(deque)


        # we cannot __setitem__ telescope lookup values, so we store them
        # in non-trait private values
        self._has_dragon_reference = {}
        self._dragon_reference_time = {}
        self._dragon_reference_counter = {}

        for tel_id in self.subarray.tel:
            self._has_dragon_reference[tel_id] = (
                self.dragon_reference_time.tel[tel_id] is not None
                and self.dragon_reference_counter.tel[tel_id] is not None
            )

            if self._has_dragon_reference[tel_id]:
                self._dragon_reference_time[tel_id] = self.dragon_reference_time.tel[tel_id]
                self._dragon_reference_counter[tel_id] = self.dragon_reference_counter.tel[tel_id]

    def __call__(self, tel_id, event):
        lst = event.lst.tel[tel_id]

        # data comes in random module order, svc contains actual order
        module_index = np.where(lst.svc.module_ids == self.dragon_module_id.tel[tel_id])[0][0]
        ucts_available = bool(lst.evt.extdevices_presence & 2)

        ucts_timestamp = lst.evt.ucts_timestamp
        ucts_time = ucts_timestamp * 1e-9

        # first event and values not passed
        if not self._has_dragon_reference[tel_id]:
            self._dragon_reference_counter[tel_id] = combine_counters(
                lst.evt.pps_counter[module_index],
                lst.evt.tenMHz_counter[module_index]
            )
            if not ucts_available:
                self.log.warning(
                    f'Cannot calculate a precise timestamp for obs_id={event.index.obs_id}'
                    f', tel_id={tel_id}. UCTS unavailable.'
                )
                # convert runstart from UTC to tai
                run_start = Time(lst.svc.date, format='unix')
                self._dragon_reference_time[tel_id] = int(1e9 * run_start.unix_tai)
            else:
                source = 'ucts'
                self._dragon_reference_time[tel_id] = ucts_timestamp
                if event.index.event_id != 1:
                    self.log.warning(
                        'Calculating time reference values not from first event.'
                        ' This might result in wrong timestamps due to UCTS jumps'
                    )

            self.log.critical(
                f'Using event {event.index.event_id} as time reference for dragon.'
                f' timestamp: {self._dragon_reference_time[tel_id]} from {source}'
                f' counter: {self._dragon_reference_counter[tel_id]}'
            )

            self._has_dragon_reference[tel_id] = True


        # Dragon/TIB timestamps based on a valid absolute reference UCTS timestamp
        dragon_timestamp = calc_dragon_time(
            lst, module_index,
            reference_time=self._dragon_reference_time[tel_id],
            reference_counter=self._dragon_reference_counter[tel_id],
        )

        # if ucts is not available, there is nothing more we have to do
        # and dragon time is our only option
        if not ucts_available:
            return time_from_unix_tai_ns(dragon_timestamp)

        # Due to a DAQ bug, sometimes there are 'jumps' in the
        # UCTS info in the raw files. After one such jump,
        # all the UCTS info attached to an event actually
        # corresponds to the next event. This one-event
        # shift stays like that until there is another jump
        # (then it becomes a 2-event shift and so on). We will
        # keep track of those jumps, by storing the UCTS info
        # of the previously read events in the list
        # previous_ucts_time_unix. The list has one element
        # for each of the jumps, so if there has been just
        # one jump we have the UCTS info of the previous
        # event only (which truly corresponds to the
        # current event). If there have been n jumps, we keep
        # the past n events. The info to be used for
        # the current event is always the first element of
        # the array, previous_ucts_time_unix[0], whereas the
        # current event's (wrong) ucts info is placed last in
        # the array. Each time the first array element is
        # used, it is removed and the rest move up in the
        # list. We have another similar array for the trigger
        # types, previous_ucts_trigger_type
        ucts_trigger_type = lst.evt.ucts_trigger_type

        if len(self.previous_ucts_timestamps[tel_id]) > 0:
            # put the current values last in the queue, for later use:
            self.previous_ucts_timestamps[tel_id].append(ucts_timestamp)
            self.previous_ucts_trigger_types[tel_id].append(ucts_trigger_type)

            # get the correct time for the current event from the queue
            ucts_timestamp = self.previous_ucts_timestamps[tel_id].popleft()
            ucts_trigger_type = self.previous_ucts_trigger_types[tel_id].popleft()

            lst.evt.ucts_trigger_type = ucts_trigger_type
            lst.evt.ucts_timestamp = ucts_timestamp

        # Now check consistency of UCTS and Dragon times. If
        # UCTS time is ahead of Dragon time by more than
        # 1 us, most likely the UCTS info has been
        # lost for this event (i.e. there has been another
        # 'jump' of those described above), and the one we have
        # actually corresponds to the next event. So we put it
        # back first in the list, to assign it to the next
        # event. We also move the other elements down in the
        # list,  which will now be one element longer.
        # We leave the current event with the same time,
        # which will be approximately correct (depending on
        # event rate), and set its ucts_trigger_type to -1,
        # which will tell us a jump happened and hence this
        # event does not have proper UCTS info.
        if (ucts_timestamp - dragon_timestamp) > 1e3:
            self.log.warning(
                f'Found UCTS jump in event {event.index.event_id}'
                f', dragon time: {dragon_timestamp:.07f}'
                f', delta: {(ucts_timestamp - dragon_timestamp):.1f} µs'
            )
            self.previous_ucts_timestamps[tel_id].appendleft(ucts_timestamp)
            self.previous_ucts_trigger_types[tel_id].appendleft(ucts_trigger_type)

            # fall back to dragon time / tib trigger
            lst.evt.ucts_timestamp = dragon_timestamp
            ucts_timestamp = dragon_timestamp

            tib_available = lst.evt.extdevices_presence & 1
            if tib_available:
                lst.evt.ucts_trigger_type = lst.evt.tib_masked_trigger
            else:
                self.log.warning(
                    'Detected ucts jump but not tib trigger info available'
                    ', event will have no trigger information'
                )
                lst.evt.ucts_trigger_type = 0

        # Select the timestamps to be used for pointing interpolation
        if self.timestamp.tel[tel_id] == "dragon":
            return time_from_unix_tai_ns(dragon_timestamp)

        return time_from_unix_tai_ns(ucts_timestamp)
