import numpy as np
from astropy.io import fits
from numba import jit, njit

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
N_CAPACITORS_CHANNEL = 1024
# 4 drs4 channels are cascaded for each pixel
N_CAPACITORS_PIXEL = 4 * N_CAPACITORS_CHANNEL
N_ROI = 40
HIGH_GAIN = 0
LOW_GAIN = 1
LAST_RUN_WITH_OLD_FIRMWARE = 1574
CLOCK_FREQUENCY_KHZ = 133e3

# we have 8 channels per module, but only 7 are used.
N_CHANNELS_PER_MODULE = 8

# First capacitor order according Dragon v5 board data format
CHANNEL_ORDER_HIGH_GAIN = [0, 0, 1, 1, 2, 2, 3]
CHANNEL_ORDER_LOW_GAIN = [4, 4, 5, 5, 6, 6, 7]

# on which module is a pixel?
MODULE_INDEX = np.repeat(np.arange(N_MODULES), 7)

CHANNEL_INDEX_LOW_GAIN = MODULE_INDEX * N_CHANNELS_PER_MODULE + np.tile(CHANNEL_ORDER_LOW_GAIN, N_MODULES)
CHANNEL_INDEX_HIGH_GAIN = MODULE_INDEX * N_CHANNELS_PER_MODULE + np.tile(CHANNEL_ORDER_HIGH_GAIN, N_MODULES)


def get_first_capacitor_for_modules(first_capacitor_id, expected_pixel_id=None):
    '''
    Get the first capacitor for each module's pixels from the
    flat first_capacitor_id array.
    '''

    # reorder if provided with correct pixel order
    if expected_pixel_id is not None:
        # expected_pixel_id is the inverse lookup of what is needed here,
        # so we create an empty array first and index into it.
        index_low_gain = np.empty_like(CHANNEL_INDEX_LOW_GAIN)
        index_low_gain[expected_pixel_id] = CHANNEL_INDEX_LOW_GAIN
        index_high_gain = np.empty_like(CHANNEL_INDEX_HIGH_GAIN)
        index_high_gain[expected_pixel_id] = CHANNEL_INDEX_HIGH_GAIN
    else:
        index_low_gain = CHANNEL_INDEX_LOW_GAIN
        index_high_gain = CHANNEL_INDEX_HIGH_GAIN

    # first: reshape so we can access by module
    fc = np.zeros((N_GAINS, N_PIXELS), dtype='uint16')
    fc[LOW_GAIN] = first_capacitor_id[index_low_gain]
    fc[HIGH_GAIN] = first_capacitor_id[index_high_gain]

    return fc


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
            shape = (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
            self.last_readout_time[tel_id] = np.zeros(shape)

            shape = (N_GAINS, N_PIXELS)
            self.first_cap[tel_id] = np.zeros(shape, dtype=int)
            self.first_cap_old[tel_id] = np.zeros(shape, dtype=int)

        self.gain_selector = GainSelector.from_name(
            self.gain_selector_type, parent=self
        )

        if self.calibration_path is not None:
            self.mon_data = self._read_calibration_file(self.calibration_path)

    def calibrate(self, event):
        for tel_id, r0 in event.r0.tel.items():
            r1 = event.r1.tel[tel_id]

            # update first caps
            self.first_cap_old[tel_id][:] = self.first_cap[tel_id][:]
            self.first_cap[tel_id] = get_first_capacitor_for_modules(
                event.lst.tel[tel_id].evt.first_capacitor_id,
                event.lst.tel[tel_id].svc.pixel_ids,
            )

            self.subtract_pedestal(event, tel_id)
            self.time_lapse_corr(event, tel_id)
            self.interpolate_spikes(event, tel_id)

            start = self.r1_sample_start.tel[tel_id]
            end = self.r1_sample_end.tel[tel_id]
            waveform = r1.waveform[..., start:end]

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
            (N_GAINS, N_PIXELS_PER_MODULE * N_MODULES, N_CAPACITORS_PIXEL + N_ROI),
            dtype=np.int16
        )
        with fits.open(path) as f:
            pedestal_data[:, :, :N_CAPACITORS_PIXEL] = f[1].data

        pedestal_data[:, :, N_CAPACITORS_PIXEL:] = pedestal_data[:, :, :N_ROI]

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
        expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids

        samples = event.r0.tel[tel_id].waveform.astype(np.float32)
        samples = subtract_pedestal_jit(
            samples,
            expected_pixel_id,
            self.first_cap[tel_id],
            self._get_drs4_pedestal_data(
                self.drs4_pedestal_path.tel[tel_id],
                offset=self.offset.tel[tel_id],
            ),
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
        local_clock_counter = event.lst.tel[tel_id].evt.local_clock_counter

        # If R1 container exists, update it inplace
        if isinstance(event.r1.tel[tel_id].waveform, np.ndarray):
            container = event.r1.tel[tel_id]
        else:
            # Modify R0 container. This is to create pedestal files.
            container = event.r0.tel[tel_id]

        waveform = container.waveform.copy()

        # We have 2 functions: one for data from 2018/10/10 to 2019/11/04 and
        # one for data from 2019/11/05 (from Run 1574) after update firmware.
        # The old readout (before 2019/11/05) is shifted by 1 cell.
        if run_id > LAST_RUN_WITH_OLD_FIRMWARE:
            time_laps_corr = do_time_lapse_corr
        else:
            time_laps_corr = do_time_lapse_corr_data_from_20181010_to_20191104

        time_laps_corr(
            waveform,
            local_clock_counter,
            self.first_cap[tel_id],
            self.last_readout_time[tel_id],
        )

        container.waveform = waveform

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

        # Interpolate spikes should be done after pedestal subtraction and time lapse correction.
        if isinstance(event.r1.tel[tel_id].waveform, np.ndarray):
            waveform = event.r1.tel[tel_id].waveform.copy()

            # We have 2 functions: one for data from 2018/10/10 to 2019/11/04 and
            # one for data from 2019/11/05 (from Run 1574) after update firmware.
            # The old readout (before 2019/11/05) is shifted by 1 cell.
            if run_id > LAST_RUN_WITH_OLD_FIRMWARE:
                interpolate_pseudo_pulses = self.interpolate_pseudo_pulses
            else:
                interpolate_pseudo_pulses = self.interpolate_pseudo_pulses_data_from_20181010_to_20191104

            interpolate_pseudo_pulses(
                waveform,
                self.first_cap[tel_id],
                self.first_cap_old[tel_id],
            )
            event.r1.tel[tel_id].waveform = waveform

    @staticmethod
    @njit()
    def interpolate_pseudo_pulses(waveform, fc, fc_old):
        """
        Interpolate Spike type A. Modifies waveform in place

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
        LAST_IN_FIRST_HALF = N_CAPACITORS_CHANNEL // 2 - 1

        for gain in range(N_GAINS):
            for pixel in range(N_PIXELS):
                last_fc = fc_old[gain, pixel]
                current_fc = fc[gain, pixel]

                for k in range(4):
                    # looking for spike A first case
                    abspos = N_CAPACITORS_CHANNEL + 1 - N_ROI - 2 - last_fc + k * N_CAPACITORS_CHANNEL + N_CAPACITORS_PIXEL
                    spike_A_position = (abspos - current_fc + N_CAPACITORS_PIXEL) % N_CAPACITORS_PIXEL

                    if 2 < spike_A_position < (N_ROI - 2):
                        # The correction is only needed for even
                        # last capacitor (lc) in the first half of the
                        # DRS4 ring
                        last_capacitor = (last_fc + N_ROI - 1) % N_CAPACITORS_CHANNEL
                        if last_capacitor % 2 == 0 and last_capacitor <= LAST_IN_FIRST_HALF:
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                    # looking for spike A second case
                    abspos = N_ROI - 1 + last_fc + k * N_CAPACITORS_CHANNEL
                    spike_A_position = (abspos - current_fc + N_CAPACITORS_PIXEL) % N_CAPACITORS_PIXEL
                    if 2 < spike_A_position < (N_ROI-2):
                        # The correction is only needed for even last capacitor (lc) in the first half of the DRS4 ring
                        last_lc = last_fc + N_ROI - 1
                        if last_lc % 2 == 0 and last_lc % N_CAPACITORS_CHANNEL <= N_CAPACITORS_CHANNEL // 2 - 1:
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)

    @staticmethod
    @njit()
    def interpolate_pseudo_pulses_data_from_20181010_to_20191104(waveform, fc, fc_old):
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
        for gain in range(N_GAINS):
            for pix in range(N_PIXELS):
                for k in range(4):
                    # looking for spike A first case
                    abspos = int(size1drs - roi_size - 2 -fc_old[gain, pix] + k * size1drs + size4drs)
                    spike_A_position = int((abspos - fc[gain, pix] + size4drs) % size4drs)
                    if (spike_A_position > 2 and spike_A_position < roi_size-2):
                        # The correction is only needed for even
                        # last capacitor (lc) in the first half of the
                        # DRS4 ring
                        if ((fc_old[gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[gain, pix]+ (roi_size-1)) % size1drs <= size1drs//2-2):
                            interpolate_spike_A(waveform, gain, spike_A_position, pix)

                    # looking for spike A second case
                    abspos = int(roi_size - 2 + fc_old[gain, pix]+ k * size1drs)
                    spike_A_position = int((abspos -fc[gain, pix] + size4drs) % size4drs)
                    if (spike_A_position > 2 and spike_A_position < (roi_size-2)):
                        # The correction is only needed for even last capacitor (lc) in the
                        # first half of the DRS4 ring
                        if ((fc_old[gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-2):
                            interpolate_spike_A(waveform, gain, spike_A_position, pix)
        return waveform


@njit()
def subtract_pedestal_jit(
    event_waveform,
    expected_pixel_id,
    first_capacitors,
    pedestal_value_array,
    n_modules
):
    """
    Numba function to subtract the drs4 pedestal.

    Creates a new waveform array with the pedestal subtracted.
    """
    waveform = np.zeros(event_waveform.shape)

    for gain in range(N_GAINS):
        for pixel_index in range(N_PIXELS):
            # waveform is already reordered to pixel ids,
            # the first caps are not, so we need to translate here.
            pixel_id = expected_pixel_id[pixel_index]

            first_cap = first_capacitors[gain, pixel_index]

            pedestal = pedestal_value_array[gain, pixel_id, first_cap:first_cap + N_ROI]
            waveform[gain, pixel_id] = event_waveform[gain, pixel_id] - pedestal
    return waveform


@njit()
def do_time_lapse_corr(
    waveform,
    local_clock_counter,
    fc,
    last_time_array,
):
    """
    Numba function for time lapse baseline correction.
    Change waveform array.
    """
    for module in range(N_MODULES):
        time_now = local_clock_counter[module]

        for gain in range(N_GAINS):
            for pixel_in_module in range(N_PIXELS_PER_MODULE):
                pixel_index = module * N_PIXELS_PER_MODULE + pixel_in_module
                first_capacitor = fc[gain, pixel_index]

                for sample in range(N_ROI):
                    capacitor = (first_capacitor + sample) % N_CAPACITORS_PIXEL

                    # apply correction if last readout available
                    if last_time_array[gain, pixel_index, capacitor] > 0:
                        time_diff = time_now - last_time_array[gain, pixel_index, capacitor]
                        time_diff_ms = time_diff / CLOCK_FREQUENCY_KHZ

                        # FIXME: Why only for values < 100 ms, negligible otherwise?
                        if time_diff_ms < 100:
                            waveform[gain, pixel_index, sample] -= ped_time(time_diff_ms)

                    # update the last read time
                    last_time_array[gain, pixel_index, capacitor] = time_now

                # now the magic of Dragon,
                # extra conditions on the number of capacitor times being updated
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                # largely refactored by M. Nöthe
                if pixel_in_module % 2 == 0:
                    first_capacitor_in_channel = first_capacitor % N_CAPACITORS_CHANNEL
                    if 767 < first_capacitor_in_channel < 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL
                        end = start + 12
                        for capacitor in range(start, end):
                            last_time_array[gain, pixel_index, capacitor % N_CAPACITORS_PIXEL] = time_now

                    elif first_capacitor_in_channel >= 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL
                        channel = first_capacitor // N_CAPACITORS_CHANNEL
                        end = (channel + 2) * N_CAPACITORS_CHANNEL
                        for capacitor in range(start, end):
                            last_time_array[gain, pixel_index, capacitor % N_CAPACITORS_PIXEL] = time_now


@jit()
def do_time_lapse_corr_data_from_20181010_to_20191104(
    waveform,
    local_clock_counter,
    fc,
    last_time_array,
):
    """
    Numba function for time lapse baseline correction.
    This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
    Change waveform array.
    """

    for module in range(N_MODULES):
        time_now = local_clock_counter[module]

        for gain in range(N_GAINS):
            for pixel_in_module in range(N_PIXELS):
                pixel_index = module * N_PIXELS_PER_MODULE + pixel_in_module
                first_capacitor = fc[gain, pixel_index]

                for sample in range(N_ROI):
                    capacitor = (first_capacitor + sample) % N_CAPACITORS_PIXEL

                    if last_time_array[gain, pixel_index, capacitor] > 0:
                        time_diff = time_now - last_time_array[gain, pixel_index, capacitor]
                        time_diff_ms = time_diff / CLOCK_FREQUENCY_KHZ

                        if time_diff_ms < 100:
                            waveform[gain, pixel_index, sample] -= ped_time(time_diff_ms)

                for sample in range(-1, N_ROI - 1):
                    last_time_array[gain, pixel_index, capacitor] = time_now

                # now the magic of Dragon,
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                # largely refactored by M. Nöthe
                if pixel_index % 2 == 0:

                    if 766 < (first_capacitor % N_CAPACITORS_CHANNEL) < 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL - 1
                        end = first_capacitor + N_CAPACITORS_CHANNEL + 11
                        for capacitor in range(start, end):
                            last_time_array[gain, pixel_index, capacitor % N_CAPACITORS_PIXEL] = time_now

                    elif first_capacitor % N_CAPACITORS_CHANNEL >= 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL
                        channel = first_capacitor // N_CAPACITORS_CHANNEL
                        end = (channel + 2) * N_CAPACITORS_CHANNEL
                        for capacitor in range(start, end):
                            last_time_array[gain, pixel_index, capacitor % N_CAPACITORS_PIXEL] = time_now


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
