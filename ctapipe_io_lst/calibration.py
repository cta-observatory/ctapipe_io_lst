import numpy as np
from astropy.io import fits
from numba import jit, prange

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Path, create_class_enum_trait, IntTelescopeParameter,
    TelescopeParameter
)

from ctapipe.calib.camera import GainSelector
from ctapipe.containers import MonitoringContainer
from ctapipe.io import HDF5TableReader
from functools import lru_cache
import tables

__all__ = [
    'LSTR0Corrections',
]

N_GAINS = 2
N_MODULES = 265
N_PIXELS_PER_MODULE = 7
N_PIXELS = N_MODULES * N_PIXELS_PER_MODULE
N_CAPACITORS = 1024
N_CAPACITORS_4 = 4 * N_CAPACITORS
N_ROI = 40
HIGH_GAIN = 0
LOW_GAIN = 1
LAST_RUN_WITH_OLD_FIRMWARE = 1574


class LSTR0Corrections(TelescopeComponent):
    """
    The base R0-level calibrator. Changes the r0 container.

    The R0 calibrator performs the camera-specific R0 calibration that is
    usually performed on the raw data by the camera server.
    This calibrator exists in lstchain for testing and prototyping purposes.
    """
    offset = IntTelescopeParameter(
        default_value=400,
        help='Define the offset of the baseline'
    ).tag(config=True)

    r1_sample_start = IntTelescopeParameter(
        default_value=3,
        help='Start sample for r1 waveform',
        allow_none=True,
    ).tag(config=True)

    r1_sample_end = IntTelescopeParameter(
        default_value=39,
        help='End sample for r1 waveform',
        allow_none=True,
    ).tag(config=True)

    drs4_pedestal_path = TelescopeParameter(
        trait=Path(exists=True, directory_ok=False),
        allow_none=True,
        default_value=None,
        help='Path to the LST pedestal file',
    ).tag(config=True)

    calibration_path = Path(
        exists=True, directory_ok=False,
        help='Path to LST calibration file',
    ).tag(config=True)

    gain_selector_type = create_class_enum_trait(
        GainSelector, default_value='ThresholdGainSelector'
    )

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        The R0 calibrator for LST data.
        Fill the r1 container.

        Parameters
        ----------
        """
        super().__init__(
            subarray=subarray, config=config, parent=parent, **kwargs
        )

        self.mon_data = None
        self.last_readout_time = {}
        self.first_cap = {}
        self.first_cap_time_lapse = {}
        self.first_cap_spikes = {}
        self.first_cap_old = {}

        for tel_id in self.subarray.tel:
            shape = (N_MODULES, N_GAINS, N_PIXELS_PER_MODULE, N_CAPACITORS)
            self.last_readout_time[tel_id] = np.zeros(shape)

            shape = (N_MODULES, N_GAINS, N_PIXELS_PER_MODULE)
            self.first_cap[tel_id] = np.zeros(shape, dtype=int)
            self.first_cap_old[tel_id] = np.zeros(shape, dtype=int)
            self.first_cap_time_lapse[tel_id] = np.zeros(shape)
            self.first_cap_spikes[tel_id] = np.zeros(shape)

        self.gain_selector = GainSelector.from_name(
            self.gain_selector_type, parent=self
        )

    def calibrate(self, event):
        for tel_id, r0 in event.r0.tel.items():
            r1 = event.r1.tel[tel_id]

            self.subtract_pedestal(event, tel_id)
            self.time_lapse_corr(event, tel_id)
            self.interpolate_spikes(event, tel_id)

            start = self.r1_sample_start.tel[tel_id]
            end = self.r1_sample_end.tel[tel_id]
            waveform = r1.waveform[..., start:end]

            # apply drs4 offset subtraction
            waveform -= self.offset.tel[tel_id]

            # apply monitoring data corrections
            if self.mon_data is not None:
                calibration = self.mon_data.tel[tel_id].calibration
                waveform -= calibration.pedestal_per_sample[:, :, np.newaxis]
                waveform *= calibration.dc_to_pe[:, :, np.newaxis]

            waveform = waveform.astype(np.float32)
            r1.selected_gain_channel = self.gain_selector(r1.waveform)
            n_pixels = waveform.shape[1]
            r1.waveform = waveform[r1.selected_gain_channel, np.arange(n_pixels)]

    @staticmethod
    def _read_calibration_file(path):
        """
        Read the correction from hdf5 calibration file
        """
        mon = MonitoringContainer()

        with tables.open_file(path) as f:
            tel_ids = [
                int(key[4:]) for key in f.root._v_children.keys()
                if key.startswith('tel_')
            ]

        for tel_id in tel_ids:
            with HDF5TableReader(path) as h5_table:
                base = f'/tel_{tel_id}'
                # read the calibration data
                table = base + '/calibration'
                next(h5_table.read(table, mon.tel[tel_id].calibration))

                # read pedestal data
                table = base + '/pedestal'
                next(h5_table.read(table, mon.tel[tel_id].pedestal))

                # read flat-field data
                table = base + '/flatfield'
                next(h5_table.read(table, mon.tel[tel_id].flatfield))

                # read the pixel_status container
                table = base + '/pixel_status'
                next(h5_table.read(table, mon.tel[tel_id].pixel_status))

        return mon

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_drs4_pedestal_data(path, offset=0):
        """
        Function to load pedestal file.

        To make boundary conditions unnecessary,
        the first N_ROI values are repeated at the end of the array

        The result is cached so we can repeatedly call this method
        using the configured path without reading it each time.
        """
        pedestal_data = np.empty(
            (N_GAINS, N_PIXELS_PER_MODULE * N_MODULES, N_CAPACITORS_4 + N_ROI),
            dtype=np.int16
        )
        with fits.open(path) as f:
            pedestal_data[:, :, :N_CAPACITORS_4] = f[1].data

        pedestal_data[:, :, N_CAPACITORS_4:] = pedestal_data[:, :, :N_ROI]

        if offset != 0:
            pedestal_data -= offset

        return pedestal_data

    def subtract_pedestal(self, event, tel_id):
        """
        Subtract cell offset using pedestal file.
        Fill the R1 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """
        n_modules = event.lst.tel[tel_id].svc.num_modules

        for nr_module in range(0, n_modules):
            self.first_cap[tel_id][nr_module, :, :] = self._get_first_capacitor(event, nr_module, tel_id)

        expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
        samples = event.r0.tel[tel_id].waveform.astype(np.float32)

        samples = subtract_pedestal_jit(
            samples,
            expected_pixel_id,
            self.first_cap[tel_id],
            self._get_drs4_pedestal_data(self.drs4_pedestal_path.tel[tel_id]),
            n_modules,
        )
        event.r1.tel[tel_id].waveform = samples[:, :, :]

    def time_lapse_corr(self, event, tel_id):
        """
        Perform time lapse baseline corrections.
        Fill the R1 container or modifies R0 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """

        run_id = event.lst.tel[tel_id].svc.configuration_id

        expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
        local_clock_list = event.lst.tel[tel_id].evt.local_clock_counter
        n_modules = event.lst.tel[tel_id].svc.num_modules
        for nr_module in range(0, n_modules):
            self.first_cap_time_lapse[tel_id][nr_module, :, :] = self._get_first_capacitor(event, nr_module, tel_id)

        # If R1 container exists, update it inplace
        if isinstance(event.r1.tel[tel_id].waveform, np.ndarray):
            samples = event.r1.tel[tel_id].waveform
        else:
            # Modify R0 container. This is to create pedestal files.
            samples = event.r0.tel[tel_id].waveform

        # We have 2 functions: one for data from 2018/10/10 to 2019/11/04 and
        # one for data from 2019/11/05 (from Run 1574) after update firmware.
        # The old readout (before 2019/11/05) is shifted by 1 cell.
        if run_id > LAST_RUN_WITH_OLD_FIRMWARE:
            time_laps_corr = do_time_lapse_corr
        else:
            time_laps_corr = do_time_lapse_corr_data_from_20181010_to_20191104

        time_laps_corr(
            samples,
            expected_pixel_id,
            local_clock_list,
            self.first_cap_time_lapse[tel_id],
            self.last_readout_time[tel_id],
            n_modules,
        )

    def interpolate_spikes(self, event, tel_id):
        """
        Interpolates spike A & B.
        Fill the R1 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """
        run_id = event.lst.tel[tel_id].svc.configuration_id

        self.first_cap_old[tel_id][:] = self.first_cap[tel_id][:]
        n_modules = event.lst.tel[tel_id].svc.num_modules

        for nr_module in range(0, n_modules):
            self.first_cap[tel_id][nr_module] = self._get_first_capacitor(event, nr_module, tel_id)

        # Interpolate spikes should be done after pedestal subtraction and time lapse correction.
        if isinstance(event.r1.tel[tel_id].waveform, np.ndarray):
            waveform = event.r1.tel[tel_id].waveform[:, :, :]
            expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids

            # We have 2 functions: one for data from 2018/10/10 to 2019/11/04 and
            # one for data from 2019/11/05 (from Run 1574) after update firmware.
            # The old readout (before 2019/11/05) is shifted by 1 cell.
            if run_id > LAST_RUN_WITH_OLD_FIRMWARE:
                interpolate_pseudo_pulses = self.interpolate_pseudo_pulses
            else:
                interpolate_pseudo_pulses = self.interpolate_pseudo_pulses_data_from_20181010_to_20191104

            event.r1.tel[tel_id].waveform = interpolate_pseudo_pulses(
                waveform,
                expected_pixel_id,
                self.first_cap_spikes[tel_id],
                self.first_cap_old[tel_id],
                n_modules,
            )

    @staticmethod
    @jit(parallel=True)
    def interpolate_pseudo_pulses(waveform, expected_pixel_id, fc, fc_old, n_modules):
        """
        Interpolate Spike A & B.
        Change waveform array.
        Parameters
        ----------
        waveform : ndarray
            Waveform stored in a numpy array of shape
            (n_gain, n_pix, n_samples).
        expected_pixel_id: ndarray
            Array stored expected pixel id
            (n_pix*n_modules).
        fc : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        fc_old : ndarray
            Value of first capacitor from previous event
            stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        n_modules : int
            Number of modules
        """
        roi_size = 40
        size1drs = 1024
        size4drs = 4096
        n_gain = 2
        n_pix = 7
        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(size1drs + 1 - roi_size - 2 - fc_old[nr_module, gain, pix] + k * size1drs + size4drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < roi_size-2):
                            # The correction is only needed for even
                            # last capacitor (lc) in the first half of the
                            # DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-1):
                                pixel = expected_pixel_id[nr_module * 7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                        # looking for spike A second case
                        abspos = int(roi_size - 1 + fc_old[nr_module, gain, pix] + k * size1drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < (roi_size-2)):
                            # The correction is only needed for even last capacitor (lc) in the
                            # first half of the DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-1):
                                pixel = expected_pixel_id[nr_module * 7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)
        return waveform

    @staticmethod
    @jit(parallel=True)
    def interpolate_pseudo_pulses_data_from_20181010_to_20191104(waveform, expected_pixel_id, fc, fc_old, n_modules):
        """
        Interpolate Spike A & B.
        This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
        Change waveform array.
        Parameters
        ----------
        waveform : ndarray
            Waveform stored in a numpy array of shape
            (n_gain, n_pix, n_samples).
        expected_pixel_id: ndarray
            Array stored expected pixel id
            (n_pix*n_modules).
        fc : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        fc_old : ndarray
            Value of first capacitor from previous event
            stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        n_modules : int
            Number of modules
        """
        roi_size = 40
        size1drs = 1024
        size4drs = 4096
        n_gain = 2
        n_pix = 7
        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(size1drs - roi_size - 2 -fc_old[nr_module, gain, pix]+ k * size1drs + size4drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix]+ size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < roi_size-2):
                            # The correction is only needed for even
                            # last capacitor (lc) in the first half of the
                            # DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix]+ (roi_size-1)) % size1drs <= size1drs//2-2):
                                pixel = expected_pixel_id[nr_module*7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                        # looking for spike A second case
                        abspos = int(roi_size - 2 + fc_old[nr_module, gain, pix]+ k * size1drs)
                        spike_A_position = int((abspos -fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < (roi_size-2)):
                            # The correction is only needed for even last capacitor (lc) in the
                            # first half of the DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-2):
                                pixel = expected_pixel_id[nr_module*7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)
        return waveform

    def _get_first_capacitor(self, event, nr_module, tel_id):
        """
        Get first capacitor values from event for nr module.
        Parameters
        ----------
        event : `ctapipe` event-container
        nr_module : number of module
        tel_id : id of the telescope
        """
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[tel_id].evt.first_capacitor_id[nr_module * 8:
                                                                 (nr_module + 1) * 8]

        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[HIGH_GAIN, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[LOW_GAIN, i] = first_cap[j]
        return fc

@jit(parallel=True)
def subtract_pedestal_jit(event_waveform, expected_pixel_id, fc_cap, pedestal_value_array, n_modules):
    """
    Numba function for subtract pedestal.
    Change waveform array.
    """
    waveform = np.zeros(event_waveform.shape)
    size4drs = 4096
    roi_size = 40
    n_gain = 2
    n_pix = 7
    for nr_module in prange(0, n_modules):
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module*7 + pix]
                position = int((fc_cap[nr_module, gain, pix]) % size4drs)
                waveform[gain, pixel, :] = \
                    (event_waveform[gain, pixel, :] -
                    pedestal_value_array[gain, pixel, position:position + roi_size])
    return waveform

@jit(parallel=True)
def do_time_lapse_corr(waveform, expected_pixel_id, local_clock_list,
                       fc, last_time_array, number_of_modules):
    """
    Numba function for time lapse baseline correction.
    Change waveform array.
    """
    size4drs = 4096
    size1drs = 1024
    roi_size = 40
    n_gain = 2
    n_pix = 7
    for nr_module in prange(0, number_of_modules):
        time_now = local_clock_list[nr_module]
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module*7 + pix]
                for k in prange(0, roi_size):
                    posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                    if last_time_array[nr_module, gain, pix, posads] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, posads]
                        time_diff_ms = time_diff / (133.e3)
                        if time_diff_ms < 100:
                            val =(waveform[gain, pixel, k] - ped_time(time_diff_ms))
                            waveform[gain, pixel, k] = val

                posads0 = int((0 + fc[nr_module, gain, pix]) % size4drs)
                if posads0+roi_size < size4drs:
                    last_time_array[nr_module, gain, pix, (posads0):(posads0+roi_size)] = time_now
                else:
                    for k in prange(0, roi_size):
                        posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                        last_time_array[nr_module, gain, pix, posads] = time_now

                # now the magic of Dragon,
                # extra conditions on the number of capacitor times being updated
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                if pix % 2 == 0:
                    first_cap = fc[nr_module, gain, pix]
                    if first_cap % size1drs > 767 and first_cap % size1drs < 1013:
                        start = int(first_cap) + size1drs
                        end = int(first_cap) + size1drs + 12
                        last_time_array[nr_module, gain, pix, (start%size4drs):(end%size4drs)] = time_now
                    elif first_cap % size1drs >= 1013:
                        channel = int(first_cap / size1drs)
                        for kk in range(first_cap + size1drs, ((channel + 2) * size1drs)):
                            last_time_array[nr_module, gain, pix, int(kk) % size4drs] = time_now

@jit(parallel=True)
def do_time_lapse_corr_data_from_20181010_to_20191104(waveform, expected_pixel_id, local_clock_list,
                                                      fc, last_time_array, number_of_modules):
    """
    Numba function for time lapse baseline correction.
    This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
    Change waveform array.
    """
    size4drs = 4096
    size1drs = 1024
    roi_size = 40
    n_gain = 2
    n_pix = 7

    for nr_module in prange(0, number_of_modules):
        time_now = local_clock_list[nr_module]
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module * 7 + pix]
                for k in prange(0, roi_size):
                    posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                    if last_time_array[nr_module, gain, pix, posads] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, posads]
                        time_diff_ms = time_diff / (133.e3)
                        if time_diff_ms < 100:
                            val = waveform[gain, pixel, k] - ped_time(time_diff_ms)
                            waveform[gain, pixel, k] = val

                posads0 = int((0 + fc[nr_module, gain, pix]) % size4drs)
                if posads0 + roi_size < size4drs and (posads0-1) > 1:
                    last_time_array[nr_module, gain, pix, (posads0-1):(posads0 + (roi_size-1))] = time_now
                else:
                    # Old firmware issue: readout shifted by 1 cell
                    for k in prange(-1, roi_size-1):
                        posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                        last_time_array[nr_module, gain, pix, posads] = time_now

                # now the magic of Dragon,
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                if pix % 2 == 0:
                    first_cap = fc[nr_module, gain, pix]
                    if first_cap % size1drs > 766 and first_cap % size1drs < 1013:
                        start = int(first_cap) + size1drs - 1
                        end = int(first_cap) + size1drs + 11
                        last_time_array[nr_module, gain, pix, (start % size4drs):(end % size4drs)] = time_now
                    elif first_cap % size1drs >= 1013:
                        channel = int(first_cap / size1drs)
                        for kk in range(first_cap + size1drs, (channel + 2) * size1drs):
                            last_time_array[nr_module, gain, pix, int(kk) % size4drs] = time_now


@jit
def ped_time(timediff):
    """
    Power law function for time lapse baseline correction.
    Coefficients from curve fitting to dragon test data
    at temperature 20 degC
    """
    # old values at 30 degC (used till release v0.4.5)
    # return 27.33 * np.power(timediff, -0.24) - 10.4

    # new values at 20 degC, provided by Yokiho Kobayashi 2/3/2020
    # see also Yokiho's talk in https://indico.cta-observatory.org/event/2664/
    return 32.99 * np.power(timediff, -0.22) - 11.9


@jit
def interpolate_spike_A(waveform, gain, position, pixel):
    """
    Numba function for interpolation spike type A.
    Change waveform array.
    """
    samples = waveform[gain, pixel, :]
    a = int(samples[position - 1])
    b = int(samples[position + 2])
    waveform[gain, pixel, position] = (samples[position - 1]) + (0.33 * (b - a))
    waveform[gain, pixel, position + 1] = (samples[position - 1]) + (0.66 * (b - a))
