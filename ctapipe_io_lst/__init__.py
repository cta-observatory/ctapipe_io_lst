# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.
"""
import numpy as np
from astropy import units as u
from pkg_resources import resource_filename
import os
from astropy.time import Time
from os import listdir
from ctapipe.core import Provenance
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    CameraDescription,
    CameraReadout,
    CameraGeometry,
    OpticsDescription,
)
from enum import IntFlag, auto

from ctapipe.io import EventSource
from ctapipe.io.datalevels import DataLevel
from ctapipe.core.traits import Int, Bool, Float, Enum, Path
from ctapipe.containers import PixelStatusContainer, EventType

from .containers import LSTArrayEventContainer, LSTServiceContainer
from .version import __version__
from .calibration import LSTR0Corrections
from .event_time import EventTimeCalculator
from .pointing import PointingSource
from .anyarray_dtypes import (
    CDTS_AFTER_37201_DTYPE,
    CDTS_BEFORE_37201_DTYPE,
    SWAT_DTYPE,
    DRAGON_COUNTERS_DTYPE,
    TIB_DTYPE,
)
from .constants import (
    HIGH_GAIN, N_PIXELS, N_SAMPLES
)

__all__ = ['LSTEventSource', '__version__']


class TriggerBits(IntFlag):
    '''
    See TIB User manual
    '''
    MONO = auto()
    STEREO = auto()
    CALIBRATION = auto()
    SINGLE_PE = auto()
    SOFTWARE = auto()
    PEDESTAL = auto()
    SLOW_CONTROL = auto()

    PHYSICS = MONO | STEREO
    OTHER = CALIBRATION | SINGLE_PE | SOFTWARE | PEDESTAL | SLOW_CONTROL


OPTICS = OpticsDescription(
    'LST',
    equivalent_focal_length=u.Quantity(28, u.m),
    num_mirrors=1,
    mirror_area=u.Quantity(386.73, u.m**2),
    num_mirror_tiles=198,
)


def get_channel_info(pixel_status):
    '''
    Extract the channel info bits from the pixel_status array.
    See R1 data model, https://forge.in2p3.fr/boards/313/topics/3033

    Returns
    -------
    channel_status: ndarray[uint8]
        0: pixel not read out (defect)
        1: high-gain read out
        2: low-gain read out
        3: both gains read out
    '''
    return (pixel_status & 0b1100) >> 2


def load_camera_geometry(version=4):
    ''' Load camera geometry from bundled resources of this repo '''
    f = resource_filename(
        'ctapipe_io_lst', f'resources/LSTCam-{version:03d}.camgeom.fits.gz'
    )
    Provenance().add_input_file(f, role="CameraGeometry")
    return CameraGeometry.from_table(f)


def read_pulse_shapes():

    '''
    Reads in the data on the pulse shapes and readout speed, from an external file

    Returns
    -------
    (daq_time_per_sample, pulse_shape_time_step, pulse shapes)
        daq_time_per_sample: time between samples in the actual DAQ (ns, astropy quantity)
        pulse_shape_time_step: time between samples in the returned single-p.e pulse shape (ns, astropy
    quantity)
        pulse shapes: Single-p.e. pulse shapes, ndarray of shape (2, 1640)
    '''

    # temporary replace the reference pulse shape
    # ("oversampled_pulse_LST_8dynode_pix6_20200204.dat")
    # with a dummy one in order to disable the charge corrections in the charge extractor
    infilename = resource_filename(
        'ctapipe_io_lst',
        'resources/oversampled_pulse_LST_8dynode_pix6_20200204.dat'
    )

    data = np.genfromtxt(infilename, dtype='float', comments='#')
    Provenance().add_input_file(infilename, role="PulseShapes")
    daq_time_per_sample = data[0, 0] * u.ns
    pulse_shape_time_step = data[0, 1] * u.ns

    # Note we have to transpose the pulse shapes array to provide what ctapipe
    # expects:
    return daq_time_per_sample, pulse_shape_time_step, data[1:].T


class LSTEventSource(EventSource):
    """EventSource for LST r0 data."""

    n_gains = Int(
        2,
        help='Number of gains at r0 level'
    ).tag(config=True)

    multi_streams = Bool(
        True,
        help='Read in parallel all streams '
    ).tag(config=True)

    min_flatfield_adc = Float(
        default_value=3000.0,
        help=(
            'Events with that have more than ``min_flatfield_pixel_fraction``'
            ' of the pixels inside [``min_flatfield_adc``, ``max_flatfield_adc``]'
            ' get tagged as EventType.FLATFIELD'
        ),
    ).tag(config=True)

    max_flatfield_adc = Float(
        default_value=12000.0,
        help=(
            'Events with that have more than ``min_flatfield_pixel_fraction``'
            ' of the pixels inside [``min_flatfield_adc``, ``max_flatfield_adc``]'
            ' get tagged as EventType.FLATFIELD'
        ),
    ).tag(config=True)

    min_flatfield_pixel_fraction = Float(
        default_value=0.8,
        help=(
            'Events with that have more than ``min_flatfield_pixel_fraction``'
            ' of the pixels inside [``min_flatfield_pe``, ``max_flatfield_pe``]'
            ' get tagged as EventType.FLATFIELD'
        ),
    ).tag(config=True)

    default_trigger_type = Enum(
        ['ucts', 'tib'], default_value='ucts',
        help=(
            'Default source for trigger type information.'
            ' For older data, tib might be the better choice but for data newer'
            ' than 2020-06-25, ucts is the preferred option. The source will still'
            ' fallback to the other device if the chosen default device is not '
            ' available'
        )
    ).tag(config=True)

    calibrate_flatfields_and_pedestals = Bool(
        default_value=True,
        help='To be set to True for calibration processing'
    ).tag(config=True)

    classes = [PointingSource, EventTimeCalculator, LSTR0Corrections]

    def __init__(self, input_url=None, **kwargs):
        """
        Parameters
        ----------
        n_gains = number of gains expected in input file

        multi_streams = enable the reading of input files from all streams

        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.\
        kwargs: dict
            Additional parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """
        super().__init__(input_url=input_url, **kwargs)

        if self.multi_streams:
            # test how many streams are there:
            # file name must be [stream name]Run[all the rest]
            # All the files with the same [all the rest] are opened

            path, name = os.path.split(os.path.abspath(self.input_url))
            if 'Run' in name:
                stream, run = name.split('Run', 1)
            else:
                run = name

            ls = listdir(path)
            self.file_list = []

            for file_name in ls:
                if run in file_name:
                    full_name = os.path.join(path, file_name)
                    self.file_list.append(full_name)

        else:
            self.file_list = [self.input_url]

        self.multi_file = MultiFiles(self.file_list)
        self.geometry_version = 4

        self.camera_config = self.multi_file.camera_config
        self.log.info(
            "Read {} input files".format(
                self.multi_file.num_inputs()
            )
        )
        self.tel_id = self.camera_config.telescope_id
        self._subarray = self.create_subarray(self.geometry_version, self.tel_id)
        self.r0_r1_calibrator = LSTR0Corrections(
            subarray=self._subarray, parent=self
        )
        self.time_calculator = EventTimeCalculator(
            subarray=self.subarray,
            run_id=self.camera_config.configuration_id,
            expected_modules_id=self.camera_config.lstcam.expected_modules_id,
            parent=self,
        )
        self.pointing_source = PointingSource(subarray=self.subarray, parent=self)
        self.lst_service = self.fill_lst_service_container(self.tel_id, self.camera_config)

    @property
    def subarray(self):
        return self._subarray

    @property
    def is_simulation(self):
        return False

    @property
    def obs_ids(self):
        # currently no obs id is available from the input files
        return [self.camera_config.configuration_id, ]

    @property
    def datalevels(self):
        if self.r0_r1_calibrator.calibration_path is not None:
            return (DataLevel.R0, DataLevel.R1)
        return (DataLevel.R0, )

    def rewind(self):
        self.multi_file.rewind()

    @staticmethod
    def create_subarray(geometry_version, tel_id=1):
        """
        Obtain the subarray from the EventSource
        Returns
        -------
        ctapipe.instrument.SubarrayDescription
        """

        # camera info from LSTCam-[geometry_version].camgeom.fits.gz file
        camera_geom = load_camera_geometry(version=geometry_version)

        # get info on the camera readout:
        daq_time_per_sample, pulse_shape_time_step, pulse_shapes = read_pulse_shapes()

        camera_readout = CameraReadout(
            'LSTCam',
            1 / daq_time_per_sample,
            pulse_shapes,
            pulse_shape_time_step,
        )

        camera = CameraDescription('LSTCam', camera_geom, camera_readout)

        lst_tel_descr = TelescopeDescription(
            name='LST', tel_type='LST', optics=OPTICS, camera=camera
        )

        tel_descriptions = {tel_id: lst_tel_descr}

        # LSTs telescope position taken from MC from the moment
        tel_positions = {tel_id: [50., 50., 16] * u.m}

        subarray = SubarrayDescription(
            name=f"LST-{tel_id} subarray",
            tel_descriptions=tel_descriptions,
            tel_positions=tel_positions,
        )

        return subarray

    def _generator(self):

        # container for LST data
        array_event = LSTArrayEventContainer()
        array_event.meta['input_url'] = self.input_url
        array_event.meta['max_events'] = self.max_events
        array_event.meta['origin'] = 'LSTCAM'

        # also add service container to the event section
        array_event.lst.tel[self.tel_id].svc = self.lst_service

        # initialize general monitoring container
        self.initialize_mon_container(array_event)

        # loop on events
        for count, zfits_event in enumerate(self.multi_file):
            array_event.count = count
            array_event.index.event_id = zfits_event.event_id
            array_event.index.obs_id = self.obs_ids[0]

            # Skip "empty" events that occur at the end of some runs
            if zfits_event.event_id == 0:
                self.log.warning('Event with event_id=0 found, skipping')
                continue

            # fill general R0 data
            self.fill_r0_container(array_event, zfits_event)

            # fill specific LST event data
            self.fill_lst_event_container(array_event, zfits_event)

            # fill trigger info (needs r0 and lst specifics)
            self.fill_trigger_info(array_event)

            # fill general monitoring data
            self.fill_mon_container(array_event, zfits_event)
            self.fill_pointing_info(array_event)

            # apply low level corrections
            if self.r0_r1_calibrator.drs4_pedestal_path.tel[self.tel_id] is not None:
                self.r0_r1_calibrator.apply_drs4_corrections(array_event)

                # flat field tagging is performed on r1 data, so can only
                # be done after the drs4 corrections are applied
                self.tag_flatfield_events(array_event)

            # gain select and calibrate to pe
            if self.r0_r1_calibrator.calibration_path is not None:

                # skip flatfield and pedestal events if asked
                if (
                    array_event.trigger.event_type not in {EventType.FLATFIELD, EventType.SKY_PEDESTAL}
                        or self.calibrate_flatfields_and_pedestals
                ):
                    self.r0_r1_calibrator.calibrate(array_event)

            yield array_event

    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: CameraConfig
            #  2: Events
            h = fits.open(file_path)[2].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        except IndexError:
            # A fits file of a different format
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'R1.CameraEvent')
        )

        is_lst_file = 'lstcam_counters' in ttypes
        return is_protobuf_zfits_file & is_lst_file

    @staticmethod
    def fill_lst_service_container(tel_id, camera_config):
        """
        Fill LSTServiceContainer with specific LST service data data
        (from the CameraConfig table of zfit file)

        """
        return LSTServiceContainer(
            telescope_id=tel_id,
            cs_serial=camera_config.cs_serial,
            configuration_id=camera_config.configuration_id,
            date=camera_config.date,
            num_pixels=camera_config.num_pixels,
            num_samples=camera_config.num_samples,
            pixel_ids=camera_config.expected_pixels_id,
            data_model_version=camera_config.data_model_version,
            num_modules=camera_config.lstcam.num_modules,
            module_ids=camera_config.lstcam.expected_modules_id,
            idaq_version=camera_config.lstcam.idaq_version,
            cdhs_version=camera_config.lstcam.cdhs_version,
            algorithms=camera_config.lstcam.algorithms,
            pre_proc_algorithms=camera_config.lstcam.pre_proc_algorithms,
        )

    def fill_lst_event_container(self, array_event, zfits_event):
        """
        Fill LSTEventContainer with specific LST service data
        (from the Event table of zfit file)

        """
        tel_id = self.tel_id

        lst_evt = array_event.lst.tel[tel_id].evt

        lst_evt.configuration_id = zfits_event.configuration_id
        lst_evt.event_id = zfits_event.event_id
        lst_evt.tel_event_id = zfits_event.tel_event_id
        lst_evt.pixel_status = zfits_event.pixel_status
        lst_evt.ped_id = zfits_event.ped_id
        lst_evt.module_status = zfits_event.lstcam.module_status
        lst_evt.extdevices_presence = zfits_event.lstcam.extdevices_presence

        # if TIB data are there
        if lst_evt.extdevices_presence & 1:
            tib = zfits_event.lstcam.tib_data.view(TIB_DTYPE)[0]
            lst_evt.tib_event_counter = tib['event_counter']
            lst_evt.tib_pps_counter = tib['pps_counter']
            lst_evt.tib_tenMHz_counter = tib['tenMHz_counter']
            lst_evt.tib_stereo_pattern = tib['stereo_pattern']
            lst_evt.tib_masked_trigger = tib['masked_trigger']

        # if UCTS data are there
        if lst_evt.extdevices_presence & 2:
            if int(array_event.lst.tel[tel_id].svc.idaq_version) > 37201:
                cdts = zfits_event.lstcam.cdts_data.view(CDTS_AFTER_37201_DTYPE)[0]
                lst_evt.ucts_timestamp = cdts[0]
                lst_evt.ucts_address = cdts[1]        # new
                lst_evt.ucts_event_counter = cdts[2]
                lst_evt.ucts_busy_counter = cdts[3]   # new
                lst_evt.ucts_pps_counter = cdts[4]
                lst_evt.ucts_clock_counter = cdts[5]
                lst_evt.ucts_trigger_type = cdts[6]
                lst_evt.ucts_white_rabbit_status = cdts[7]
                lst_evt.ucts_stereo_pattern = cdts[8] # new
                lst_evt.ucts_num_in_bunch = cdts[9]   # new
                lst_evt.ucts_cdts_version = cdts[10]  # new

            else:
                # unpack UCTS-CDTS data (old version)
                cdts = zfits_event.lstcam.cdts_data.view(CDTS_BEFORE_37201_DTYPE)[0]
                lst_evt.ucts_event_counter = cdts[0]
                lst_evt.ucts_pps_counter = cdts[1]
                lst_evt.ucts_clock_counter = cdts[2]
                lst_evt.ucts_timestamp = cdts[3]
                lst_evt.ucts_camera_timestamp = cdts[4]
                lst_evt.ucts_trigger_type = cdts[5]
                lst_evt.ucts_white_rabbit_status = cdts[6]

        # if SWAT data are there
        if lst_evt.extdevices_presence & 4:
            # unpack SWAT data
            unpacked_swat = zfits_event.lstcam.swat_data.view(SWAT_DTYPE)[0]
            lst_evt.swat_timestamp = unpacked_swat[0]
            lst_evt.swat_counter1 = unpacked_swat[1]
            lst_evt.swat_counter2 = unpacked_swat[2]
            lst_evt.swat_event_type = unpacked_swat[3]
            lst_evt.swat_camera_flag = unpacked_swat[4]
            lst_evt.swat_camera_event_num = unpacked_swat[5]
            lst_evt.swat_array_flag = unpacked_swat[6]
            lst_evt.swat_array_event_num = unpacked_swat[7]

        # unpack Dragon counters
        counters = zfits_event.lstcam.counters.view(DRAGON_COUNTERS_DTYPE)
        lst_evt.pps_counter = counters['pps_counter']
        lst_evt.tenMHz_counter = counters['tenMHz_counter']
        lst_evt.event_counter = counters['event_counter']
        lst_evt.trigger_counter = counters['trigger_counter']
        lst_evt.local_clock_counter = counters['local_clock_counter']

        lst_evt.chips_flags = zfits_event.lstcam.chips_flags
        lst_evt.first_capacitor_id = zfits_event.lstcam.first_capacitor_id
        lst_evt.drs_tag_status = zfits_event.lstcam.drs_tag_status
        lst_evt.drs_tag = zfits_event.lstcam.drs_tag

    def fill_trigger_info(self, array_event):
        tel_id = self.tel_id

        trigger = array_event.trigger
        trigger.time = self.time_calculator(tel_id, array_event)
        trigger.tels_with_trigger = [tel_id]
        trigger.tel[tel_id].time = trigger.time

        lst = array_event.lst.tel[tel_id]
        tib_available = lst.evt.extdevices_presence & 1
        ucts_available = lst.evt.extdevices_presence & 2

        # decide which source to use, if both are available,
        # the option decides, if not, fallback to the avilable source
        # if no source available, warn and do not fill trigger info
        if tib_available and ucts_available:
            if self.default_trigger_type == 'ucts':
                trigger_bits = lst.evt.ucts_trigger_type
            else:
                trigger_bits = lst.evt.tib_masked_trigger

        elif tib_available:
            trigger_bits = lst.evt.tib_masked_trigger

        elif ucts_available:
            trigger_bits = lst.evt.ucts_trigger_type

        else:
            self.log.warning('No trigger info available.')
            trigger.event_type = EventType.UNKNOWN
            return

        if (
            ucts_available
            and lst.evt.ucts_trigger_type == 42
            and self.default_trigger_type == "ucts"
        ) :
            self.log.warning(
                'Event with UCTS trigger_type 42 found.'
                ' Probably means unreliable or shifted UCTS data.'
                ' Consider switching to TIB using `default_trigger_type="tib"`'
            )

        # first bit mono trigger, second stereo.
        # If *only* those two are set, we assume it's a physics event
        # for all other we only check if the flag is present
        if (trigger_bits & TriggerBits.PHYSICS) and not (trigger_bits & TriggerBits.OTHER):
            trigger.event_type = EventType.SUBARRAY
        elif trigger_bits & TriggerBits.CALIBRATION:
            trigger.event_type = EventType.FLATFIELD
        elif trigger_bits & TriggerBits.PEDESTAL:
            trigger.event_type = EventType.SKY_PEDESTAL
        elif trigger_bits & TriggerBits.SINGLE_PE:
            trigger.event_type = EventType.SINGLE_PE
        else:
            self.log.warning(f'Event {array_event.index.event_id} has unknown event type, trigger: {trigger_bits:08b}')
            trigger.event_type = EventType.UNKNOWN

    def tag_flatfield_events(self, array_event):
        '''
        Use a heuristic based on R1 waveforms to recognize flat field events

        Currently, tagging of flat field events does not work,
        they are reported as physics events, here a heuristic identifies
        those events. Since trigger types might be wrong due to ucts errors,
        we try to identify flat field events in all trigger types.

        DRS4 corrections but not the p.e. calibration must be applied
        '''
        tel_id = self.tel_id

        image = array_event.r1.tel[tel_id].waveform[HIGH_GAIN].sum(axis=1)
        in_range = (image >= self.min_flatfield_adc) & (image <= self.max_flatfield_adc)

        if np.count_nonzero(in_range) >= self.min_flatfield_pixel_fraction * image.size:
            self.log.debug(
                'Setting event type of event'
                f' {array_event.index.event_id} to FLATFIELD'
            )
            array_event.trigger.event_type = EventType.FLATFIELD

    def fill_pointing_info(self, array_event):
        tel_id = self.tel_id
        # for now, make filling pointing info optional,
        # only do it when a drive report has been given.
        if self.pointing_source.drive_report_path.tel[tel_id] is not None:

            pointing = self.pointing_source.get_pointing_position_altaz(
                tel_id, array_event.trigger.time,
            )
            array_event.pointing.tel[tel_id] = pointing
            array_event.pointing.array_altitude = pointing.altitude
            array_event.pointing.array_azimuth = pointing.azimuth

            ra, dec = self.pointing_source.get_pointing_position_icrs(
                tel_id, array_event.trigger.time
            )
            array_event.pointing.array_ra = ra
            array_event.pointing.array_dec = dec

        elif array_event.count == 0:
            # but make a warning on the first event if it is missing
            self.log.warning(
                'No drive report specified, pointing info will not be filled'
            )

    def fill_r0_camera_container(self, r0_container, zfits_event):
        """
        Fill with R0CameraContainer
        """

        try:
            reshaped_waveform = zfits_event.waveform.reshape(
                self.n_gains,
                self.camera_config.num_pixels,
                self.camera_config.num_samples
            )
        except ValueError:
            raise ValueError(
                f"Number of gains not correct, waveform shape is {zfits_event.waveform.shape[0]}"
                f" instead of "
                f"{self.camera_config.num_pixels * self.camera_config.num_samples * self.n_gains}"
            )

        # re-order the waveform following the expected_pixels_id values
        #  could also just do waveform = reshaped_waveform[np.argsort(expected_ids)]
        dtype = reshaped_waveform.dtype
        fill = np.iinfo(dtype).max
        reordered_waveform = np.full((self.n_gains, N_PIXELS, N_SAMPLES), fill, dtype=dtype)
        reordered_waveform[:, self.camera_config.expected_pixels_id, :] = reshaped_waveform
        r0_container.waveform = reordered_waveform

    def fill_r0_container(self, array_event, zfits_event):
        """
        Fill with R0Container

        """
        container = array_event.r0

        r0_camera_container = container.tel[self.tel_id]
        self.fill_r0_camera_container(
            r0_camera_container,
            zfits_event
        )

    def initialize_mon_container(self, array_event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        container = array_event.mon
        mon_camera_container = container.tel[self.tel_id]

        shape = (self.n_gains, N_PIXELS)
        # all pixels broken by default
        status_container = PixelStatusContainer(
            hardware_failing_pixels=np.ones(shape, dtype=bool),
            pedestal_failing_pixels=np.zeros(shape, dtype=bool),
            flatfield_failing_pixels=np.zeros(shape, dtype=bool),
        )
        mon_camera_container.pixel_status = status_container

    def fill_mon_container(self, array_event, zfits_event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        status_container = array_event.mon.tel[self.tel_id].pixel_status

        # reorder the array
        expected_pixels_id = self.camera_config.expected_pixels_id

        reordered_pixel_status = np.zeros(N_PIXELS, dtype=zfits_event.pixel_status.dtype)
        reordered_pixel_status[expected_pixels_id] = zfits_event.pixel_status

        channel_info = get_channel_info(reordered_pixel_status)
        status_container.hardware_failing_pixels[:] = channel_info == 0


class MultiFiles:

    """
    This class open all the files in file_list and read the events following
    the event_id order
    """

    def __init__(self, file_list):
        file_list = list(file_list)

        if len(file_list) == 0:
            raise ValueError('`file_list` must not be empty')

        self._file = {}
        self._events = {}
        self._events_table = {}
        self._camera_config = {}
        self.camera_config = None

        paths = []
        for file_name in file_list:
            paths.append(file_name)
            Provenance().add_input_file(file_name, role='r0.sub.evt')

        # open the files and get the first fits Tables
        from protozfits import File

        for path in paths:

            try:
                self._file[path] = File(str(path))
                self._events_table[path] = self._file[path].Events
                self._events[path] = next(self._file[path].Events)

                # verify where the CameraConfig is present
                if 'CameraConfig' in self._file[path].__dict__.keys():
                    self._camera_config[path] = next(self._file[path].CameraConfig)

                # for the moment it takes the first CameraConfig it finds (to be changed)
                    if self.camera_config is None:
                        self.camera_config = self._camera_config[path]

            except StopIteration:
                pass

        # verify that somewhere the CameraConfing is present
        if self.camera_config is None:
            raise IOError(f"No CameraConfig was found in any of the input files: {paths}")

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_event()

    def next_event(self):
        # check for the minimal event id
        if not self._events:
            raise StopIteration

        min_path = min(
            self._events.items(),
            key=lambda item: item[1].event_id,
        )[0]

        # return the minimal event id
        next_event = self._events[min_path]
        try:
            self._events[min_path] = next(self._file[min_path].Events)
        except StopIteration:
            del self._events[min_path]

        return next_event

    def __len__(self):
        total_length = sum(
            len(table)
            for table in self._events_table.values()
        )
        return total_length

    def rewind(self):
        for name, file in self._file.items():
            file.Events.protobuf_i_fits.rewind()

    def num_inputs(self):
        return len(self._file)
