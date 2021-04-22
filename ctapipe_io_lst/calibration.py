from functools import lru_cache

import numpy as np
from astropy.io import fits
import astropy.units as u
from numba import njit
import tables

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Path, IntTelescopeParameter,
    TelescopeParameter, FloatTelescopeParameter, Bool, Float
)

from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.containers import MonitoringContainer, ArrayEventContainer
from ctapipe.io import HDF5TableReader


from .constants import (
    N_GAINS, N_PIXELS, N_MODULES, N_SAMPLES, LOW_GAIN, HIGH_GAIN,
    N_PIXELS_MODULE, N_CAPACITORS_PIXEL, N_CAPACITORS_CHANNEL,
    LAST_RUN_WITH_OLD_FIRMWARE, CLOCK_FREQUENCY_KHZ,
    CHANNEL_ORDER_LOW_GAIN, CHANNEL_ORDER_HIGH_GAIN, N_CHANNELS_MODULE
)

__all__ = [
    'LSTR0Corrections',
]


@lru_cache()
def pixel_channel_indices(n_modules):
    module_index = np.repeat(np.arange(n_modules), 7)
    low_gain = module_index * N_CHANNELS_MODULE + np.tile(CHANNEL_ORDER_LOW_GAIN, n_modules)
    high_gain = module_index * N_CHANNELS_MODULE + np.tile(CHANNEL_ORDER_HIGH_GAIN, n_modules)
    return low_gain, high_gain


def get_first_capacitors_for_pixels(first_capacitor_id, expected_pixel_id=None):
    '''
    Get the first capacitor for each pixel / gain

    Parameters
    ----------
    first_capacitor_id: np.ndarray
        First capacitor array as delivered by the event builder,
        containing first capacitors for each DRS4 chip.
    expected_pixel_id: np.ndarray
        Array of the pixel ids corresponding to the positions in
        the data array.
        If given, will be used to reorder the start cells to pixel id order.

    Returns
    -------
    fc: np.ndarray
        First capacitors for each pixel in each gain, shape (N_GAINS, N_PIXELS)
    '''

    fc = np.zeros((N_GAINS, N_PIXELS), dtype='uint16')

    n_modules = first_capacitor_id.size // N_CHANNELS_MODULE

    low_gain_channels, high_gain_channels = pixel_channel_indices(n_modules)
    low_gain = first_capacitor_id[low_gain_channels]
    high_gain = first_capacitor_id[high_gain_channels]

    if expected_pixel_id is None:
        fc[LOW_GAIN] = low_gain
        fc[HIGH_GAIN] = high_gain
    else:
        fc[LOW_GAIN, expected_pixel_id] = low_gain
        fc[HIGH_GAIN, expected_pixel_id] = high_gain

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

    drs4_time_calibration_path = TelescopeParameter(
        trait=Path(exists=True, directory_ok=False),
        help='Path to the time calibration file',
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    calib_scale_high_gain = FloatTelescopeParameter(
        default_value=1.18,
        help='High gain waveform is multiplied by this number'
    ).tag(config=True)

    calib_scale_low_gain = FloatTelescopeParameter(
        default_value=1.09,
        help='Low gain waveform is multiplied by this number'
    ).tag(config=True)

    select_gain = Bool(
        default_value=True,
        help='Set to False to keep both gains.'
    ).tag(config=True)

    add_calibration_timeshift = Bool(
        default_value=True,
        help=(
            'If true, time correction from the calibration'
            ' file is added to calibration.dl1.time'
        ),
    ).tag(config=True)

    gain_selection_threshold = Float(
        default_value=3500,
        help='Threshold for the ThresholdGainSelector.'
    ).tag(config=True)

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
        self.first_cap_old = {}
        self.fbn = {}
        self.fan = {}

        for tel_id in self.subarray.tel:
            shape = (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
            self.last_readout_time[tel_id] = np.zeros(shape, dtype='uint64')

            shape = (N_GAINS, N_PIXELS)
            self.first_cap[tel_id] = np.zeros(shape, dtype=int)
            self.first_cap_old[tel_id] = np.zeros(shape, dtype=int)

        # set the right default for our default selector, change back afterwards
        # to not impact other code.
        if self.select_gain:
            self.gain_selector = ThresholdGainSelector(
                threshold=self.gain_selection_threshold,
                parent=self
            )
        else:
            self.gain_selector = None

        if self.calibration_path is not None:
            self.mon_data = self._read_calibration_file(self.calibration_path)

    def apply_drs4_corrections(self, event: ArrayEventContainer):
        self.update_first_capacitors(event)

        for tel_id in event.r0.tel:
            r1 = event.r1.tel[tel_id]
            # fill r1 waveform with a copy of the r0 converted to float32
            # float32 can represent all values of uint16 exactly, so this
            # does not loose precision.
            r1.waveform = event.r0.tel[tel_id].waveform.astype(np.float32)

            # apply drs4 corrections
            self.subtract_pedestal(event, tel_id)
            self.time_lapse_corr(event, tel_id)
            self.interpolate_spikes(event, tel_id)

            # remove samples at beginning / end of waveform
            start = self.r1_sample_start.tel[tel_id]
            end = self.r1_sample_end.tel[tel_id]
            r1.waveform = r1.waveform[..., start:end]
            waveform = r1.waveform

            waveform -= self.offset.tel[tel_id]

            mon = event.mon.tel[tel_id]
            waveform[mon.pixel_status.hardware_failing_pixels] = 0.0

    def update_first_capacitors(self, event: ArrayEventContainer):
        for tel_id in event.r0.tel:
            lst = event.lst.tel[tel_id]
            self.first_cap_old[tel_id] = self.first_cap[tel_id]
            self.first_cap[tel_id] = get_first_capacitors_for_pixels(
                lst.evt.first_capacitor_id,
                lst.svc.pixel_ids,
            )

    def calibrate(self, event: ArrayEventContainer):

        for tel_id in event.r0.tel:
            r1 = event.r1.tel[tel_id]
            waveform = r1.waveform

            # do gain selection before converting to pe
            # like eventbuilder will do
            if self.select_gain:
                selected_gain_channel = self.gain_selector(waveform)
            else:
                selected_gain_channel = None

            # apply monitoring data corrections,
            # subtract pedestal and convert to pe
            if self.mon_data is not None:
                calibration = self.mon_data.tel[tel_id].calibration
                waveform -= calibration.pedestal_per_sample[:, :, np.newaxis]
                waveform *= calibration.dc_to_pe[:, :, np.newaxis]

            mon = event.mon.tel[tel_id]
            waveform[mon.pixel_status.hardware_failing_pixels] = 0.0

            waveform = waveform.astype(np.float32)
            n_gains, n_pixels, n_samples = waveform.shape
            pixel_index = np.arange(n_pixels)

            if selected_gain_channel is not None:
                r1.waveform = waveform[selected_gain_channel, pixel_index]
                r1.selected_gain_channel = selected_gain_channel
            else:
                r1.waveform = waveform
                r1.selected_gain_channel = None

            # store calibration data needed for dl1 calibration in ctapipe
            # first drs4 time shift (zeros if no calib file was given)
            time_shift = self.get_drs4_time_correction(
                tel_id, self.first_cap[tel_id],
                selected_gain_channel=selected_gain_channel,
            )

            # time shift from flat fielding
            if self.mon_data is not None and self.add_calibration_timeshift:
                time_corr = self.mon_data.tel[tel_id].calibration.time_correction
                # time_shift is subtracted in ctapipe,
                # but time_correction should be added
                if selected_gain_channel is not None:
                    time_shift -= time_corr[r1.selected_gain_channel, pixel_index].to_value(u.ns)
                else:
                    time_shift -= time_corr.to_value(u.ns)

            event.calibration.tel[tel_id].dl1.time_shift = time_shift

            # needed for charge scaling in ctpaipe dl1 calib
            if selected_gain_channel is not None:
                relative_factor = np.empty(n_pixels)
                relative_factor[r1.selected_gain_channel == HIGH_GAIN] = self.calib_scale_high_gain.tel[tel_id]
                relative_factor[r1.selected_gain_channel == LOW_GAIN] = self.calib_scale_low_gain.tel[tel_id]
            else:
                relative_factor = np.empty((n_gains, n_pixels))
                relative_factor[HIGH_GAIN] = self.calib_scale_high_gain.tel[tel_id]
                relative_factor[LOW_GAIN] = self.calib_scale_low_gain.tel[tel_id]

            event.calibration.tel[tel_id].dl1.relative_factor = relative_factor

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
    def load_drs4_time_calibration_file(path):
        """
        Function to load calibration file.
        """
        with tables.open_file(path, 'r') as f:
            fan = f.root.fan[:]
            fbn = f.root.fbn[:]

        return fan, fbn

    def load_drs4_time_calibration_file_for_tel(self, tel_id):
        self.fan[tel_id], self.fbn[tel_id] = self.load_drs4_time_calibration_file(
            self.drs4_time_calibration_path.tel[tel_id]
        )

    def get_drs4_time_correction(self, tel_id, first_capacitors, selected_gain_channel=None):
        """
        Return pulse time after time correction.
        """

        if self.drs4_time_calibration_path.tel[tel_id] is None:
            if selected_gain_channel is None:
                return np.zeros((N_GAINS, N_PIXELS))
            else:
                return np.zeros(N_PIXELS)

        # load calib file if not already done
        if tel_id not in self.fan:
            self.load_drs4_time_calibration_file_for_tel(tel_id)

        if selected_gain_channel is not None:
            return calc_drs4_time_correction_gain_selected(
                first_capacitors,
                selected_gain_channel,
                self.fan[tel_id],
                self.fbn[tel_id],
            )
        else:
            return calc_drs4_time_correction_both_gains(
                first_capacitors,
                self.fan[tel_id],
                self.fbn[tel_id],
            )

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_drs4_pedestal_data(path, offset=0):
        """
        Function to load pedestal file.

        To make boundary conditions unnecessary,
        the first N_SAMPLES values are repeated at the end of the array

        The result is cached so we can repeatedly call this method
        using the configured path without reading it each time.
        """
        pedestal_data = np.empty(
            (N_GAINS, N_PIXELS_MODULE * N_MODULES, N_CAPACITORS_PIXEL + N_SAMPLES),
            dtype=np.int16
        )
        with fits.open(path) as f:
            pedestal_data[:, :, :N_CAPACITORS_PIXEL] = f[1].data

        pedestal_data[:, :, N_CAPACITORS_PIXEL:] = pedestal_data[:, :, :N_SAMPLES]

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
        subtract_pedestal_jit(
            event.r1.tel[tel_id].waveform,
            self.first_cap[tel_id],
            self._get_drs4_pedestal_data(
                self.drs4_pedestal_path.tel[tel_id],
                offset=self.offset.tel[tel_id],
            ),
        )

    def time_lapse_corr(self, event, tel_id):
        """
        Perform time lapse baseline corrections.
        Fill the R1 container or modifies R0 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """
        lst = event.lst.tel[tel_id]

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
        run_id = event.lst.tel[tel_id].svc.configuration_id
        if run_id > LAST_RUN_WITH_OLD_FIRMWARE:
            time_lapse_corr = do_time_lapse_corr
        else:
            time_lapse_corr = do_time_lapse_corr_data_from_20181010_to_20191104

        time_lapse_corr(
            waveform,
            lst.evt.local_clock_counter,
            self.first_cap[tel_id],
            self.last_readout_time[tel_id],
            lst.svc.pixel_ids,
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
    @njit(cache=True)
    def interpolate_pseudo_pulses(waveform, fc, fc_old):
        """
        Interpolate Spike type A. Modifies waveform in place

        Parameters
        ----------
        waveform : ndarray
            Waveform stored in a numpy array of shape
            (N_GAINS, N_PIXELS, N_SAMPLES).
        fc : ndarray
            Value of first capacitor stored in a numpy array of shape
            (N_GAINS, N_PIXELS).
        fc_old : ndarray
            Value of first capacitor from previous event
            stored in a numpy array of shape
            (N_GAINS, N_PIXELS).
        """
        LAST_IN_FIRST_HALF = N_CAPACITORS_CHANNEL // 2 - 1

        for gain in range(N_GAINS):
            for pixel in range(N_PIXELS):
                last_fc = fc_old[gain, pixel]
                current_fc = fc[gain, pixel]

                for k in range(4):
                    # looking for spike A first case
                    abspos = N_CAPACITORS_CHANNEL + 1 - N_SAMPLES - 2 - last_fc + k * N_CAPACITORS_CHANNEL + N_CAPACITORS_PIXEL
                    spike_A_position = (abspos - current_fc + N_CAPACITORS_PIXEL) % N_CAPACITORS_PIXEL

                    if 2 < spike_A_position < (N_SAMPLES - 2):
                        # The correction is only needed for even
                        # last capacitor (lc) in the first half of the
                        # DRS4 ring
                        last_capacitor = (last_fc + N_SAMPLES - 1) % N_CAPACITORS_CHANNEL
                        if last_capacitor % 2 == 0 and last_capacitor <= LAST_IN_FIRST_HALF:
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                    # looking for spike A second case
                    abspos = N_SAMPLES - 1 + last_fc + k * N_CAPACITORS_CHANNEL
                    spike_A_position = (abspos - current_fc + N_CAPACITORS_PIXEL) % N_CAPACITORS_PIXEL
                    if 2 < spike_A_position < (N_SAMPLES-2):
                        # The correction is only needed for even last capacitor (lc) in the first half of the DRS4 ring
                        last_lc = last_fc + N_SAMPLES - 1
                        if last_lc % 2 == 0 and last_lc % N_CAPACITORS_CHANNEL <= N_CAPACITORS_CHANNEL // 2 - 1:
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)

    @staticmethod
    @njit(cache=True)
    def interpolate_pseudo_pulses_data_from_20181010_to_20191104(waveform, fc, fc_old):
        """
        Interpolate Spike A
        This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
        Change waveform array.

        Parameters
        ----------
        waveform : ndarray
            Waveform stored in a numpy array of shape
            (N_GAINS, N_PIXELS, N_SAMPLES).
        fc : ndarray
            Value of first capacitor stored in a numpy array of shape
            (N_GAINS, N_PIXELS).
        fc_old : ndarray
            Value of first capacitor from previous event
            stored in a numpy array of shape
            (N_GAINS, N_PIXELS).
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


@njit(cache=True)
def subtract_pedestal_jit(
    waveform,
    first_capacitors,
    pedestal_value_array,
):
    """
    Numba function to subtract the drs4 pedestal.
    Mutates input array inplace
    """

    for gain in range(N_GAINS):
        for pixel_id in range(N_PIXELS):
            # waveform is already reordered to pixel ids,
            # the first caps are not, so we need to translate here.
            first_cap = first_capacitors[gain, pixel_id]
            pedestal = pedestal_value_array[gain, pixel_id, first_cap:first_cap + N_SAMPLES]
            waveform[gain, pixel_id] -= pedestal

@njit(cache=True)
def do_time_lapse_corr(
    waveform,
    local_clock_counter,
    first_capacitors,
    last_readout_time,
    expected_pixels_id,
):
    """
    Numba function for time lapse baseline correction.
    Change waveform array.
    """
    n_modules = len(expected_pixels_id) // N_PIXELS_MODULE
    for gain in range(N_GAINS):
        for module in range(n_modules):
            time_now = local_clock_counter[module]

            for pixel_in_module in range(N_PIXELS_MODULE):
                pixel_index = module * N_PIXELS_MODULE + pixel_in_module
                pixel_id = expected_pixels_id[pixel_index]
                first_capacitor = first_capacitors[gain, pixel_id]

                for sample in range(N_SAMPLES):
                    capacitor = (first_capacitor + sample) % N_CAPACITORS_PIXEL

                    last_readout_time_cap = last_readout_time[gain, pixel_id, capacitor]

                    # apply correction if last readout available
                    if last_readout_time_cap > 0:
                        time_diff = time_now - last_readout_time_cap
                        time_diff_ms = time_diff / CLOCK_FREQUENCY_KHZ

                        # FIXME: Why only for values < 100 ms, negligible otherwise?
                        if time_diff_ms < 100:
                            # prevent underflow of the unsigned int value
                            correction = min(ped_time(time_diff_ms), waveform[gain, pixel_id, sample])
                            waveform[gain, pixel_id, sample] -= correction

                    # update the last read time
                    last_readout_time[gain, pixel_id, capacitor] = time_now

                # now the magic of Dragon,
                # extra conditions on the number of capacitor times being updated
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                # largely refactored by M. Nöthe
                if (pixel_in_module % 2) == 0:
                    first_capacitor_in_channel = first_capacitor % N_CAPACITORS_CHANNEL
                    if (first_capacitor_in_channel > 767) and (first_capacitor_in_channel < 1013):
                        start = first_capacitor + N_CAPACITORS_CHANNEL
                        end = start + 12

                        for capacitor in range(start, end):
                            last_readout_time[gain, pixel_id, capacitor % N_CAPACITORS_PIXEL] = time_now

                    elif first_capacitor_in_channel >= 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL
                        channel = first_capacitor // N_CAPACITORS_CHANNEL
                        end = (channel + 2) * N_CAPACITORS_CHANNEL
                        for capacitor in range(start, end):
                            last_readout_time[gain, pixel_id, capacitor % N_CAPACITORS_PIXEL] = time_now


@njit(cache=True)
def do_time_lapse_corr_data_from_20181010_to_20191104(
    waveform,
    local_clock_counter,
    first_capacitors,
    last_readout_time,
    expected_pixels_id,
):
    """
    Numba function for time lapse baseline correction.
    This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
    Change waveform array.
    """

    n_modules = len(expected_pixels_id) // N_PIXELS_MODULE
    for gain in range(N_GAINS):
        for module in range(n_modules):
            time_now = local_clock_counter[module]

            for pixel_in_module in range(N_PIXELS):
                pixel_index = module * N_PIXELS_MODULE + pixel_in_module
                pixel_id = expected_pixels_id[pixel_index]

                first_capacitor = first_capacitors[gain, pixel_id]

                for sample in range(N_SAMPLES):
                    capacitor = (first_capacitor + sample) % N_CAPACITORS_PIXEL

                    if last_readout_time[gain, pixel_id, capacitor] > 0:
                        time_diff = time_now - last_readout_time[gain, pixel_id, capacitor]
                        time_diff_ms = time_diff / CLOCK_FREQUENCY_KHZ

                        if time_diff_ms < 100:
                            # prevent underflow of the unsigned int value
                            correction = min(ped_time(time_diff_ms), waveform[gain, pixel_id, sample])
                            waveform[gain, pixel_id, sample] -= correction

                for sample in range(-1, N_SAMPLES - 1):
                    capacitor = (first_capacitor + sample) % N_CAPACITORS_PIXEL
                    last_readout_time[gain, pixel_id, capacitor] = time_now

                # now the magic of Dragon,
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                # largely refactored by M. Nöthe
                if pixel_id % 2 == 0:

                    if 766 < (first_capacitor % N_CAPACITORS_CHANNEL) < 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL - 1
                        end = first_capacitor + N_CAPACITORS_CHANNEL + 11
                        for capacitor in range(start, end):
                            last_readout_time[gain, pixel_id, capacitor % N_CAPACITORS_PIXEL] = time_now

                    elif first_capacitor % N_CAPACITORS_CHANNEL >= 1013:
                        start = first_capacitor + N_CAPACITORS_CHANNEL
                        channel = first_capacitor // N_CAPACITORS_CHANNEL
                        end = (channel + 2) * N_CAPACITORS_CHANNEL
                        for capacitor in range(start, end):
                            last_readout_time[gain, pixel_id, capacitor % N_CAPACITORS_PIXEL] = time_now


@njit(cache=True)
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
    return 32.99 * timediff**(-0.22) - 11.9


@njit(cache=True)
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


@njit(cache=True)
def calc_drs4_time_correction_gain_selected(
    first_capacitors, selected_gain_channel, fan, fbn
):
    _n_gains, n_pixels, n_harmonics = fan.shape
    time = np.zeros(n_pixels)

    for pixel in range(n_pixels):
        gain = selected_gain_channel[pixel]
        first_capacitor = first_capacitors[gain, pixel]
        time[pixel] = calc_fourier_time_correction(
            first_capacitor, fan[gain, pixel], fbn[gain, pixel]
        )
    return time


@njit(cache=True)
def calc_drs4_time_correction_both_gains(
    first_capacitors, fan, fbn
):
    time = np.zeros((N_GAINS, N_PIXELS))

    for gain in range(N_GAINS):
        for pixel in range(N_PIXELS):
            first_capacitor = first_capacitors[gain, pixel]
            time[gain, pixel] = calc_fourier_time_correction(
                first_capacitor, fan[gain, pixel], fbn[gain, pixel]
            )
    return time


@njit(cache=True)
def calc_fourier_time_correction(first_capacitor, fan, fbn):
    n_harmonics = len(fan)

    time = 0
    first_capacitor = first_capacitor % N_CAPACITORS_CHANNEL

    for harmonic in range(1, n_harmonics):
        a = fan[harmonic]
        b = fbn[harmonic]
        omega = harmonic * (2 * np.pi / N_CAPACITORS_CHANNEL)

        time += a * np.cos(omega * first_capacitor)
        time += b * np.sin(omega * first_capacitor)

    return time
