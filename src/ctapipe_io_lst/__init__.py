# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.
"""
from ctapipe.instrument.subarray import EarthLocation
import numpy as np
from astropy import units as u
from pkg_resources import resource_filename
from ctapipe.core import Provenance
from ctapipe.instrument import (
    ReflectorShape,
    TelescopeDescription,
    SubarrayDescription,
    CameraDescription,
    CameraReadout,
    CameraGeometry,
    OpticsDescription,
    SizeType,
)
from enum import IntFlag, auto
from astropy.time import Time

from ctapipe.io import EventSource, read_table
from ctapipe.io.datalevels import DataLevel
from ctapipe.core.traits import Bool, Float, Enum, Path
from ctapipe.containers import (
    CoordinateFrameType, PixelStatusContainer, EventType, PointingMode, R0CameraContainer, R1CameraContainer,
    SchedulingBlockContainer, ObservationBlockContainer,
)
from ctapipe.coordinates import CameraFrame

from ctapipe_io_lst.ground_frame import ground_frame_from_earth_location

from .multifiles import MultiFiles
from .containers import LSTArrayEventContainer, LSTServiceContainer, LSTEventContainer
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
    HIGH_GAIN, LST_LOCATIONS, N_GAINS, N_PIXELS, N_SAMPLES, LST1_LOCATION, REFERENCE_LOCATION,
)


__all__ = ['LSTEventSource', '__version__']


# Date from which the flatfield heuristic will be switch off by default
NO_FF_HEURISTIC_DATE = Time("2022-01-01T00:00:00")


class TriggerBits(IntFlag):
    '''
    See TIB User manual
    '''
    UNKNOWN = 0
    MONO = auto()
    STEREO = auto()
    CALIBRATION = auto()
    SINGLE_PE = auto()
    SOFTWARE = auto()
    PEDESTAL = auto()
    SLOW_CONTROL = auto()

    PHYSICS = MONO | STEREO
    OTHER = CALIBRATION | SINGLE_PE | SOFTWARE | PEDESTAL | SLOW_CONTROL


class PixelStatus(IntFlag):
    '''
    Pixel status information

    See Section A.5 of the CTA R1 Data Model:
    https://forge.in2p3.fr/dmsf/files/8627
    '''
    RESERVED_0 = auto()
    RESERVED_1 = auto()
    HIGH_GAIN_STORED = auto()
    LOW_GAIN_STORED = auto()
    SATURATED = auto()
    PIXEL_TRIGGER_1 = auto()
    PIXEL_TRIGGER_2 = auto()
    PIXEL_TRIGGER_3 = auto()

    BOTH_GAINS_STORED = HIGH_GAIN_STORED | LOW_GAIN_STORED

#: LST Optics Description
OPTICS = OpticsDescription(
    name='LST',
    size_type=SizeType.LST,
    n_mirrors=1,
    n_mirror_tiles=198,
    reflector_shape=ReflectorShape.PARABOLIC,
    equivalent_focal_length=u.Quantity(28, u.m),
    effective_focal_length=u.Quantity(29.30565, u.m),
    mirror_area=u.Quantity(386.73, u.m**2),
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


def load_camera_geometry():
    ''' Load camera geometry from bundled resources of this repo '''
    f = resource_filename(
        'ctapipe_io_lst', 'resources/LSTCam.camgeom.fits.gz'
    )
    Provenance().add_input_file(f, role="CameraGeometry")
    cam = CameraGeometry.from_table(f)
    cam.frame = CameraFrame(focal_length=OPTICS.effective_focal_length)
    return cam


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
    """
    EventSource for LST R0 data.
    """
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

    use_flatfield_heuristic = Bool(
        default_value=None,
        allow_none=True,
        help=(
            'Whether or not to try to identify flat field events independent of'
            ' the trigger type in the event. If None (the default) the decision'
            ' will be made based on the date of the run, as this should only be'
            ' needed for data from before 2022, when a TIB firmware update fixed'
            ' the issue with unreliable UCTS information in the event data'
        ),
    ).tag(config=True)

    calibrate_flatfields_and_pedestals = Bool(
        default_value=True,
        help='If True, flat field and pedestal events are also calibrated.'
    ).tag(config=True)

    apply_drs4_corrections = Bool(
        default_value=True,
        help=(
            'Apply DRS4 corrections.'
            ' If True, this will fill R1 waveforms with the corrections applied'
            ' Use the options for the LSTR0Corrections to configure which'
            ' corrections are applied'
        ),
    ).tag(config=True)

    trigger_information = Bool(
        default_value=True,
        help='Fill trigger information.'
    ).tag(config=True)

    pointing_information = Bool(
        default_value=True,
        help=(
            'Fill pointing information.'
            ' Requires specifying `PointingSource.drive_report_path`'
        ),
    ).tag(config=True)

    pedestal_ids_path = Path(
        default_value=None,
        exists=True,
        allow_none=True,
        help=(
            'Path to a file containing the ids of the interleaved pedestal events'
            ' for the current input file'
        )
    ).tag(config=True)

    reference_position_lon = Float(
        default_value=REFERENCE_LOCATION.lon.deg,
        help=(
            "Longitude of the reference location for telescope GroundFrame coordinates."
            " Default is the roughly area weighted average of LST-1, MAGIC-1 and MAGIC-2."
        )
    ).tag(config=True)

    reference_position_lat = Float(
        default_value=REFERENCE_LOCATION.lat.deg,
        help=(
            "Latitude of the reference location for telescope GroundFrame coordinates."
            " Default is the roughly area weighted average of LST-1, MAGIC-1 and MAGIC-2."
        )
    ).tag(config=True)

    reference_position_height = Float(
        default_value=REFERENCE_LOCATION.height.to_value(u.m),
        help=(
            "Height of the reference location for telescope GroundFrame coordinates."
            " Default is current MC obslevel."
        )
    ).tag(config=True)


    classes = [PointingSource, EventTimeCalculator, LSTR0Corrections]

    def __init__(self, input_url=None, **kwargs):
        '''
        Create a new LSTEventSource.

        If the input file follows LST naming schemes, the source will
        look for related files in the same directory, depending on them
        ``all_streams`` an ``all_subruns`` options.

        if ``all_streams`` is True and the file has stream=1, then the
        source will also look for all other available streams and iterate
        events ordered by ``event_id``. 

        if ``all_subruns`` is True and the file has subrun=0, then the
        source will also look for all other available subruns and read all 
        of them.

        Parameters
        ----------
        input_url: Path
            Path or url understood by ``ctapipe.core.traits.Path``.
        **kwargs:
            Any of the traitlets. See ``LSTEventSource.class_print_help``
        '''
        super().__init__(input_url=input_url, **kwargs)

        self.multi_file = MultiFiles(
            self.input_url,
            parent=self,
        )
        self.camera_config = self.multi_file.camera_config
        self.run_id = self.camera_config.configuration_id
        self.tel_id = self.camera_config.telescope_id
        self.run_start = Time(self.camera_config.date, format='unix')

        reference_location = EarthLocation(
            lon=self.reference_position_lon * u.deg,
            lat=self.reference_position_lat * u.deg,
            height=self.reference_position_height * u.m,
        )
        self._subarray = self.create_subarray(self.tel_id, reference_location)
        self.r0_r1_calibrator = LSTR0Corrections(
            subarray=self._subarray, parent=self
        )
        self.time_calculator = EventTimeCalculator(
            subarray=self.subarray,
            run_id=self.run_id,
            expected_modules_id=self.camera_config.lstcam.expected_modules_id,
            parent=self,
        )
        self.pointing_source = PointingSource(subarray=self.subarray, parent=self)
        self.lst_service = self.fill_lst_service_container(self.tel_id, self.camera_config)

        target_info = {}
        pointing_mode = PointingMode.UNKNOWN
        if self.pointing_information:
            target = self.pointing_source.get_target(tel_id=self.tel_id, time=self.run_start)
            if target is not None:
                target_info["subarray_pointing_lon"] = target["ra"]
                target_info["subarray_pointing_lat"] = target["dec"]
                target_info["subarray_pointing_frame"] = CoordinateFrameType.ICRS
                pointing_mode = PointingMode.TRACK

        self._scheduling_blocks = {
            self.run_id: SchedulingBlockContainer(
                sb_id=np.uint64(self.run_id),
                producer_id=f"LST-{self.tel_id}",
                pointing_mode=pointing_mode,
            )
        }

        self._observation_blocks = {
            self.run_id: ObservationBlockContainer(
                obs_id=np.uint64(self.run_id),
                sb_id=np.uint64(self.run_id),
                producer_id=f"LST-{self.tel_id}",
                actual_start_time=self.run_start,
                **target_info
            )
        }

        self.read_pedestal_ids()



        if self.use_flatfield_heuristic is None:
            self.use_flatfield_heuristic = self.run_start < NO_FF_HEURISTIC_DATE
            self.log.info(f"Changed `use_flatfield_heuristic` to {self.use_flatfield_heuristic}")

    @property
    def subarray(self):
        return self._subarray

    @property
    def is_simulation(self):
        return False

    @property
    def obs_ids(self):
        # currently no obs id is available from the input files
        return list(self.observation_blocks)

    @property
    def observation_blocks(self):
        return self._observation_blocks

    @property
    def scheduling_blocks(self):
        return self._scheduling_blocks

    @property
    def datalevels(self):
        if self.r0_r1_calibrator.calibration_path is not None:
            return (DataLevel.R0, DataLevel.R1)
        return (DataLevel.R0, )

    @staticmethod
    def create_subarray(tel_id=1, reference_location=None):
        """
        Obtain the subarray from the EventSource
        Returns
        -------
        ctapipe.instrument.SubarrayDescription
        """
        if reference_location is None:
            reference_location = REFERENCE_LOCATION

        camera_geom = load_camera_geometry()

        # get info on the camera readout:
        daq_time_per_sample, pulse_shape_time_step, pulse_shapes = read_pulse_shapes()

        camera_readout = CameraReadout(
            name='LSTCam',
            n_pixels=N_PIXELS,
            n_channels=N_GAINS,
            n_samples=N_SAMPLES,
            sampling_rate=(1 / daq_time_per_sample).to(u.GHz),
            reference_pulse_shape=pulse_shapes,
            reference_pulse_sample_width=pulse_shape_time_step,
        )

        camera = CameraDescription(name='LSTCam', geometry=camera_geom, readout=camera_readout)

        lst_tel_descr = TelescopeDescription(
            name='LST', optics=OPTICS, camera=camera
        )

        tel_descriptions = {tel_id: lst_tel_descr}

        xyz = ground_frame_from_earth_location(
            LST_LOCATIONS[tel_id],
            reference_location,
        ).cartesian.xyz
        tel_positions = {tel_id: xyz}

        subarray = SubarrayDescription(
            name=f"LST-{tel_id} subarray",
            tel_descriptions=tel_descriptions,
            tel_positions=tel_positions,
            reference_location=LST1_LOCATION,
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

            self.fill_r0r1_container(array_event, zfits_event)
            self.fill_lst_event_container(array_event, zfits_event)
            if self.trigger_information:
                self.fill_trigger_info(array_event)

            self.fill_mon_container(array_event, zfits_event)

            if self.pointing_information:
                self.fill_pointing_info(array_event)

            # apply low level corrections
            if self.apply_drs4_corrections:
                self.r0_r1_calibrator.apply_drs4_corrections(array_event)

                # flat field tagging is performed on r1 data, so can only
                # be done after the drs4 corrections are applied
                if self.use_flatfield_heuristic:
                    self.tag_flatfield_events(array_event)

            if self.pedestal_ids is not None:
                self.check_interleaved_pedestal(array_event)

            # gain select and calibrate to pe
            if self.r0_r1_calibrator.calibration_path is not None:
                # skip flatfield and pedestal events if asked
                if (
                    self.calibrate_flatfields_and_pedestals
                    or array_event.trigger.event_type not in {EventType.FLATFIELD, EventType.SKY_PEDESTAL}
                ):
                    self.r0_r1_calibrator.calibrate(array_event)

            yield array_event

    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits

        try:
            with fits.open(file_path) as hdul:
                if "Events" not in hdul:
                    return False

                header = hdul["Events"].header
                ttypes = {
                    value for key, value in header.items()
                    if 'TTYPE' in key
                }
        except OSError:
            return False


        is_protobuf_zfits_file = (
            (header['XTENSION'] == 'BINTABLE')
            and (header['ZTABLE'] is True)
            and (header['ORIGIN'] == 'CTA')
            and (header['PBFHEAD'] == 'R1.CameraEvent')
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

        # create a fresh container so we are sure we have the invalid value
        # markers in case on subsystem is going missing mid of run
        lst_evt = LSTEventContainer()
        array_event.lst.tel[tel_id].evt = lst_evt

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

        lst_evt.ucts_jump = False

    @staticmethod
    def _event_type_from_trigger_bits(trigger_bits):
        # first bit mono trigger, second stereo.
        # If *only* those two are set, we assume it's a physics event
        # for all other we only check if the flag is present
        if (trigger_bits & TriggerBits.PHYSICS) and not (trigger_bits & TriggerBits.OTHER):
            return EventType.SUBARRAY

        # We only want to tag events as flatfield that *only* have the CALIBRATION bit
        # or both CALIBRATION and MONO bits, since flatfield events might 
        # trigger the physics trigger
        if trigger_bits == TriggerBits.CALIBRATION:
            return EventType.FLATFIELD

        if trigger_bits == (TriggerBits.CALIBRATION | TriggerBits.MONO):
            return EventType.FLATFIELD

        # all other event types must match exactly
        if trigger_bits == TriggerBits.PEDESTAL:
            return EventType.SKY_PEDESTAL

        if trigger_bits == TriggerBits.SINGLE_PE:
            return EventType.SINGLE_PE

        return EventType.UNKNOWN

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

        trigger.event_type = self._event_type_from_trigger_bits(trigger_bits)
        if trigger.event_type == EventType.UNKNOWN:
            self.log.warning(f'Event {array_event.index.event_id} has unknown event type, trigger: {trigger_bits:08b}')

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
        waveform = array_event.r1.tel[tel_id].waveform

        # needs to work for gain already selected or not
        if waveform.ndim == 3:
            image = waveform[HIGH_GAIN].sum(axis=1)
        else:
            image = waveform.sum(axis=1)

        in_range = (image >= self.min_flatfield_adc) & (image <= self.max_flatfield_adc)
        n_in_range = np.count_nonzero(in_range)

        looks_like_ff = n_in_range >= self.min_flatfield_pixel_fraction * image.size
        if looks_like_ff:
            array_event.trigger.event_type = EventType.FLATFIELD
            self.log.debug(
                'Setting event type of event'
                f' {array_event.index.event_id} to FLATFIELD'
            )
        elif array_event.trigger.event_type == EventType.FLATFIELD:
            self.log.warning(
                'Found FF event that does not fulfill FF criteria: %d',
                array_event.index.event_id,
            )
            array_event.trigger.event_type = EventType.UNKNOWN

    def fill_pointing_info(self, array_event):
        tel_id = self.tel_id
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

    def fill_r0r1_camera_container(self, zfits_event):
        """
        Fill the r0 or r1 container, depending on whether gain
        selection has already happened (r1) or not (r0)

        This will create waveforms of shape (N_GAINS, N_PIXELS, N_SAMPLES),
        or (N_PIXELS, N_SAMPLES) respectively regardless of the n_pixels, n_samples
        in the file.

        Missing or broken pixels are filled using maxval of the waveform dtype.
        """
        n_pixels = self.camera_config.num_pixels
        n_samples = self.camera_config.num_samples
        expected_pixels = self.camera_config.expected_pixels_id

        has_low_gain = (zfits_event.pixel_status & PixelStatus.LOW_GAIN_STORED).astype(bool)
        has_high_gain = (zfits_event.pixel_status & PixelStatus.HIGH_GAIN_STORED).astype(bool)
        broken_pixels = ~(has_low_gain | has_high_gain)

        # broken pixels have both false, so gain selected means checking
        # if there are any pixels where exactly one of high or low gain is stored
        gain_selected = np.any(has_low_gain != has_high_gain)

        # fill value for broken pixels
        dtype = zfits_event.waveform.dtype
        fill = np.iinfo(dtype).max

        # we assume that either all pixels are gain selected or none
        # only broken pixels are allowed to be missing completely
        if gain_selected:
            selected_gain = np.where(has_high_gain, 0, 1)
            waveform = zfits_event.waveform.reshape((-1, n_samples))

            # up-to-now, we have two cases how broken pixels are dealt with
            # 1. mark them as broken but data is still included
            # 2. completely removed from EVB
            # the code here works for both cases but not for the hypothetical
            # case of broken pixels marked as broken (so camera config as 1855 pixels)
            # and 1855 pixel_status entries but broken pixels not contained in `waveform`
            if np.any(broken_pixels) and len(waveform) < n_pixels:
                raise NotImplementedError(
                    "Case of broken pixels not contained in waveform is not implemented."
                    "If you encounter this error, open an issue in ctapipe_io_lst noting"
                    " the run for which this happened."
                )

            reordered_waveform = np.full((N_PIXELS, N_SAMPLES), fill, dtype=dtype)
            reordered_waveform[expected_pixels] = waveform

            reordered_selected_gain = np.full(N_PIXELS, -1, dtype=np.int8)
            reordered_selected_gain[expected_pixels] = selected_gain

            r0 = R0CameraContainer()
            r1 = R1CameraContainer(
                waveform=reordered_waveform,
                selected_gain_channel=reordered_selected_gain,
            )
        else:
            reshaped_waveform = zfits_event.waveform.reshape(N_GAINS, n_pixels, n_samples)
            # re-order the waveform following the expected_pixels_id values
            #  could also just do waveform = reshaped_waveform[np.argsort(expected_ids)]
            reordered_waveform = np.full((N_GAINS, N_PIXELS, N_SAMPLES), fill, dtype=dtype)
            reordered_waveform[:, expected_pixels, :] = reshaped_waveform
            r0 = R0CameraContainer(waveform=reordered_waveform)
            r1 = R1CameraContainer()

        return r0, r1

    def fill_r0r1_container(self, array_event, zfits_event):
        """
        Fill with R0Container

        """
        r0, r1 = self.fill_r0r1_camera_container(zfits_event)
        array_event.r0.tel[self.tel_id] = r0
        array_event.r1.tel[self.tel_id] = r1

    def initialize_mon_container(self, array_event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        container = array_event.mon
        mon_camera_container = container.tel[self.tel_id]

        shape = (N_GAINS, N_PIXELS)
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

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.multi_file.close()

    def read_pedestal_ids(self):
        if self.pedestal_ids_path is not None:
            t = read_table(self.pedestal_ids_path, '/interleaved_pedestal_ids')
            Provenance().add_input_file(
                self.pedestal_ids_path, role="InterleavedPedestalIDs"
            )
            self.pedestal_ids = set(t['event_id'])
        else:
            self.pedestal_ids = None


    def check_interleaved_pedestal(self, array_event):
        event_id = array_event.index.event_id

        if event_id in self.pedestal_ids:
            array_event.trigger.event_type = EventType.SKY_PEDESTAL
            self.log.debug("Event %d is an interleaved pedestal", event_id)

        elif array_event.trigger.event_type == EventType.SKY_PEDESTAL:
            # wrongly tagged pedestal event must be cosmic, since it would
            # have been changed to flatfield by the flatfield tagging if ff
            array_event.trigger.event_type = EventType.SUBARRAY
            self.log.debug(
                "Event %d is tagged as pedestal but not a known pedestal event",
                event_id,
            )

