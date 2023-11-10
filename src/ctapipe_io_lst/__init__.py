# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.
"""
from importlib.resources import files, as_file
from ctapipe.instrument.subarray import EarthLocation
import logging
import numpy as np
from astropy import units as u
from ctapipe import __version__ as ctapipe_version
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
from astropy.time import Time

from ctapipe.io import EventSource, read_table
from ctapipe.io.datalevels import DataLevel
from ctapipe.core.traits import Bool, Float, Enum, Path
from ctapipe.containers import (
    CoordinateFrameType, PixelStatusContainer, EventType, PointingMode, R0CameraContainer, R1CameraContainer,
    SchedulingBlockContainer, ObservationBlockContainer, MonitoringContainer, MonitoringCameraContainer,
    EventIndexContainer,
)
from ctapipe.coordinates import CameraFrame

from ctapipe_io_lst.ground_frame import ground_frame_from_earth_location

from .multifiles import MultiFiles
from .containers import LSTArrayEventContainer, LSTServiceContainer, LSTEventContainer, LSTCameraContainer
from .version import __version__
from .calibration import LSTR0Corrections
from .event_time import EventTimeCalculator, cta_high_res_to_time
from .pointing import PointingSource
from .anyarray_dtypes import (
    CDTS_AFTER_37201_DTYPE,
    CDTS_BEFORE_37201_DTYPE,
    SWAT_DTYPE,
    DRAGON_COUNTERS_DTYPE,
    TIB_DTYPE,
    parse_tib_10MHz_counter,
)
from .constants import (
    HIGH_GAIN, LST_LOCATIONS, N_GAINS, N_PIXELS, N_SAMPLES, REFERENCE_LOCATION,
    PixelStatus, TriggerBits,
)

from .evb_preprocessing import get_processings_for_trigger_bits, EVBPreprocessing


__all__ = [
    'LSTEventSource',
    '__version__',
]


CTAPIPE_VERSION = tuple(int(v) for v in ctapipe_version.split(".")[:3])
CTAPIPE_0_20 = CTAPIPE_VERSION >= (0, 20)


log = logging.getLogger(__name__)


# Date from which the flatfield heuristic will be switch off by default
NO_FF_HEURISTIC_DATE = Time("2022-01-01T00:00:00")



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
    return (pixel_status & PixelStatus.BOTH_GAINS_STORED) >> 2


def load_camera_geometry():
    ''' Load camera geometry from bundled resources of this repo '''
    with as_file(files("ctapipe_io_lst") / "resources/LSTCam.camgeom.fits.gz") as path:
        Provenance().add_input_file(path, role="CameraGeometry")
        cam = CameraGeometry.from_table(path)
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

    pulse_shape_path = 'resources/oversampled_pulse_LST_8dynode_pix6_20200204.dat'
    with as_file(files("ctapipe_io_lst") / pulse_shape_path) as path:
        data = np.genfromtxt(path, dtype='float', comments='#')
        Provenance().add_input_file(path, role="PulseShapes")

    daq_time_per_sample = data[0, 0] * u.ns
    pulse_shape_time_step = data[0, 1] * u.ns

    # Note we have to transpose the pulse shapes array to provide what ctapipe
    # expects:
    return daq_time_per_sample, pulse_shape_time_step, data[1:].T


def _reorder_pixel_status(pixel_status, pixel_id_map, set_dvr_bits=True):
    if set_dvr_bits:
        not_broken = get_channel_info(pixel_status) != 0
        pixel_status = pixel_status[not_broken] | PixelStatus.DVR_STATUS_0

    reordered_pixel_status = np.zeros(N_PIXELS, dtype=pixel_status.dtype)
    reordered_pixel_status[pixel_id_map] = pixel_status
    return reordered_pixel_status


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
        self.dvr_applied = self.multi_file.dvr_applied

        reference_location = EarthLocation(
            lon=self.reference_position_lon * u.deg,
            lat=self.reference_position_lat * u.deg,
            height=self.reference_position_height * u.m,
        )

        self.cta_r1 = self.multi_file.cta_r1
        if self.cta_r1:
            self.tel_id = self.camera_config.tel_id
            self.local_run_id = self.camera_config.local_run_id
            self.module_id_map = self.camera_config.module_id_map
            self.pixel_id_map = self.camera_config.pixel_id_map
            self.run_start = Time(self.camera_config.config_time_s, format="unix")
            self.n_pixels = self.camera_config.num_pixels
            self.n_samples = self.camera_config.num_samples_nominal
            self.lst_service = self.fill_lst_service_container_ctar1(self.camera_config)
            self.evb_preprocessing = get_processings_for_trigger_bits(self.camera_config)
            self.data_stream = self.multi_file.data_stream
        else:
            self.tel_id = self.camera_config.telescope_id
            self.local_run_id = self.camera_config.configuration_id
            self.module_id_map = self.camera_config.lstcam.expected_modules_id
            self.pixel_id_map = self.camera_config.expected_pixels_id
            self.run_start = Time(self.camera_config.date, format="unix")
            self.n_pixels = self.camera_config.num_pixels
            self.n_samples = self.camera_config.num_samples
            self.lst_service = self.fill_lst_service_container(self.tel_id, self.camera_config)
            self.evb_preprocessing = None
            self.data_stream = None

        self.reverse_pixel_id_map = np.argsort(self.pixel_id_map)

        self._subarray = self.create_subarray(self.tel_id, reference_location)
        self.r0_r1_calibrator = LSTR0Corrections(
            subarray=self._subarray, parent=self
        )
        self.time_calculator = EventTimeCalculator(
            subarray=self.subarray,
            run_id=self.local_run_id,
            expected_modules_id=self.module_id_map,
            parent=self,
        )
        self.pointing_source = PointingSource(subarray=self.subarray, parent=self)

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
            self.local_run_id: SchedulingBlockContainer(
                sb_id=np.uint64(self.local_run_id),
                producer_id=f"LST-{self.tel_id}",
                pointing_mode=pointing_mode,
            )
        }

        self._observation_blocks = {
            self.local_run_id: ObservationBlockContainer(
                obs_id=np.uint64(self.local_run_id),
                sb_id=np.uint64(self.local_run_id),
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
        if self.cta_r1:
            if EVBPreprocessing.CALIBRATION in self.evb_preprocessing[TriggerBits.MONO]:
                return (DataLevel.R1, )
            else:
                return (DataLevel.R0, )

        if self.r0_r1_calibrator.calibration_path is not None:
            return (DataLevel.R0, DataLevel.R1)
        return (DataLevel.R0, )

    @staticmethod
    def create_subarray(tel_id=1, reference_location=None):
        """
        Obtain a single-lst subarray description

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
            reference_location=reference_location,
        )

        return subarray

    def fill_from_cta_r1(self, array_event, zfits_event):
        tel_id = self.tel_id
        scale = self.data_stream.waveform_scale
        offset = self.data_stream.waveform_offset
        pixel_id_map = self.camera_config.pixel_id_map

        # FIXME: missing modules / pixels
        # FIXME: DVR? should not happen in r1 but dl0, but our own converter uses the old R1

        # reorder to nominal pixel order
        pixel_status = _reorder_pixel_status(
            zfits_event.pixel_status, pixel_id_map, set_dvr_bits=True
        )

        n_channels = zfits_event.num_channels
        n_pixels = zfits_event.num_pixels
        n_samples = zfits_event.num_samples

        readout_shape = (n_channels, n_pixels, n_samples)
        raw_waveform = zfits_event.waveform.reshape(readout_shape)
        waveform = raw_waveform.astype(np.float32) / scale - offset

        reordered_waveform = np.full((n_channels, N_PIXELS, n_samples), 0.0, dtype=np.float32)
        reordered_waveform[:, pixel_id_map] = waveform
        waveform = reordered_waveform

        # FIXME, check using evb_preprocessing and make ctapipe support 2 gains
        if zfits_event.num_channels == 2:
            selected_gain_channel = None
        else:
            has_high_gain = (pixel_status & PixelStatus.HIGH_GAIN_STORED).astype(bool)
            selected_gain_channel = np.where(has_high_gain, 0, 1)
            waveform = waveform[0]

        array_event.lst.tel[self.tel_id] = self.fill_lst_from_ctar1(zfits_event)

        trigger = array_event.trigger
        trigger.time = cta_high_res_to_time(zfits_event.event_time_s, zfits_event.event_time_qns)
        trigger.tels_with_trigger = [tel_id]
        trigger.tel[tel_id].time = trigger.time
        trigger.event_type = EventType(zfits_event.event_type)

        r1 = R1CameraContainer(
            waveform=waveform,
            selected_gain_channel=selected_gain_channel,
        )

        if CTAPIPE_0_20:
            r1.pixel_status = pixel_status
            r1.event_type = EventType(zfits_event.event_type)
            r1.event_time = trigger.time

        array_event.r1.tel[self.tel_id] = r1

        if DataLevel.R0 in self.datalevels:
            reordered_raw_waveform = np.full((n_channels, N_PIXELS, n_samples), 0, dtype=np.uint16)
            reordered_raw_waveform[:, pixel_id_map] = raw_waveform
            array_event.r0.tel[self.tel_id] = R0CameraContainer(
                waveform=reordered_raw_waveform,
            )

    def fill_lst_from_ctar1(self, zfits_event):
        evt = LSTEventContainer(
            pixel_status=zfits_event.pixel_status.copy(),
            first_capacitor_id=zfits_event.first_cell_id,
            calibration_monitoring_id=zfits_event.calibration_monitoring_id,
            local_clock_counter=zfits_event.module_hires_local_clock_counter,
        )
        # set bits for dvr if not already set
        if not self.dvr_applied:
            not_broken = get_channel_info(evt.pixel_status) != 0
            evt.pixel_status[not_broken] |= PixelStatus.DVR_STATUS_0

        if zfits_event.debug is not None:
            debug = zfits_event.debug
            evt.module_status = debug.module_status
            evt.extdevices_presence = debug.extdevices_presence
            evt.chips_flags = debug.chips_flags
            evt.charges_hg = debug.charges_gain1
            evt.charges_lg = debug.charges_gain2

            # unpack Dragon counters
            counters = debug.counters.view(DRAGON_COUNTERS_DTYPE)
            evt.pps_counter = counters['pps_counter']
            evt.tenMHz_counter = counters['tenMHz_counter']
            evt.event_counter = counters['event_counter']
            evt.trigger_counter = counters['trigger_counter']
            evt.local_clock_counter = counters['local_clock_counter']

            # if TIB data are there
            if evt.extdevices_presence & 1:
                tib = debug.tib_data.view(TIB_DTYPE)[0]
                evt.tib_event_counter = tib['event_counter']
                evt.tib_pps_counter = tib['pps_counter']
                evt.tib_tenMHz_counter = parse_tib_10MHz_counter(tib['tenMHz_counter'])
                evt.tib_stereo_pattern = tib['stereo_pattern']
                evt.tib_masked_trigger = tib['masked_trigger']

            # if UCTS data are there
            if evt.extdevices_presence & 2:
                cdts = debug.cdts_data.view(CDTS_AFTER_37201_DTYPE)[0]
                evt.ucts_timestamp = cdts["timestamp"]
                evt.ucts_address = cdts["address"]
                evt.ucts_event_counter = cdts["event_counter"]
                evt.ucts_busy_counter = cdts["busy_counter"]
                evt.ucts_pps_counter = cdts["pps_counter"]
                evt.ucts_clock_counter = cdts["clock_counter"]
                evt.ucts_trigger_type = cdts["trigger_type"]
                evt.ucts_white_rabbit_status = cdts["white_rabbit_status"]
                evt.ucts_stereo_pattern = cdts["stereo_pattern"]
                evt.ucts_num_in_bunch = cdts["num_in_bunch"]
                evt.ucts_cdts_version = cdts["cdts_version"]

            # if SWAT data are there
            if evt.extdevices_presence & 4:
                # unpack SWAT data
                swat = debug.swat_data.view(SWAT_DTYPE)[0]
                evt.swat_assigned_event_id = swat["assigned_event_id"]
                evt.swat_trigger_id = swat["trigger_id"]
                evt.swat_trigger_type = swat["trigger_type"]
                evt.swat_trigger_time_s = swat["trigger_time_s"]
                evt.swat_trigger_time_qns = swat["trigger_time_qns"]
                evt.swat_readout_requested = swat["readout_requested"]
                evt.swat_data_available = swat["data_available"]
                evt.swat_hardware_stereo_trigger_mask = swat["hardware_stereo_trigger_mask"]
                evt.swat_negative_flag = swat["negative_flag"]

        return LSTCameraContainer(evt=evt, svc=self.lst_service)

    def _generator(self):

        # also add service container to the event section

        # initialize general monitoring container
        mon = self.initialize_mon_container()

        # loop on events
        for count, zfits_event in enumerate(self.multi_file):
            # Skip "empty" events that occur at the end of some runs
            if zfits_event.event_id == 0:
                self.log.warning('Event with event_id=0 found, skipping')
                continue

            # container for LST data
            array_event = LSTArrayEventContainer(
                count=count,
                index=EventIndexContainer(
                    obs_id=self.local_run_id,
                    event_id=zfits_event.event_id,
                ),
                mon=mon,
            )
            array_event.meta['input_url'] = self.input_url
            array_event.meta['max_events'] = self.max_events
            array_event.meta['origin'] = 'LSTCAM'

            array_event.lst.tel[self.tel_id].svc = self.lst_service

            if self.cta_r1:
                self.fill_from_cta_r1(array_event, zfits_event)
            else:
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
                    log.debug("FITS file does not contain an EVENTS HDU, returning False")
                    return False

                header = hdul["Events"].header
                ttypes = {
                    value for key, value in header.items()
                    if 'TTYPE' in key
                }
        except Exception as e:
            log.debug(f"Error trying to open input file as fits: {e}")
            return False

        if header["XTENSION"] != "BINTABLE":
            log.debug(f"EVENTS HDU is not a bintable")
            return False

        if not header.get("ZTABLE", False):
            log.debug(f"ZTABLE is not in header or False")
            return False

        if header.get("ORIGIN", "") != "CTA":
            log.debug("ORIGIN != CTA")
            return False

        proto_class = header.get("PBFHEAD")
        if proto_class is None:
            log.debug("Missing PBFHEAD key")
            return False

        supported_protos = {
            "R1.CameraEvent",
            "ProtoR1.CameraEvent",
            "CTAR1.Event",
            "R1v1.Event",
        }
        if proto_class not in supported_protos:
            log.debug(f"Unsupported PBDHEAD: {proto_class}")
            return False

        # TODO: how to differentiate nectarcam from lstcam?
        return True

    @staticmethod
    def fill_lst_service_container(tel_id, camera_config):
        """
        Fill LSTServiceContainer with specific LST service data data
        (from the CameraConfig table of zfit file)

        """
        return LSTServiceContainer(
            telescope_id=tel_id,
            local_run_id=camera_config.configuration_id,
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

    @staticmethod
    def fill_lst_service_container_ctar1(camera_config):
        """
        Fill LSTServiceContainer with specific LST service data data
        (from the CameraConfig table of zfit file)

        """
        return LSTServiceContainer(
            telescope_id=camera_config.tel_id,
            local_run_id=camera_config.local_run_id,
            date=camera_config.config_time_s,
            configuration_id=camera_config.camera_config_id,
            pixel_ids=camera_config.pixel_id_map,
            module_ids=camera_config.module_id_map,
            num_modules=camera_config.num_modules,
            num_pixels=camera_config.num_pixels,
            num_channels=camera_config.num_channels,
            num_samples=camera_config.num_samples_nominal,

            data_model_version=camera_config.data_model_version,
            calibration_service_id=camera_config.calibration_service_id,
            calibration_algorithm_id=camera_config.calibration_algorithm_id,

            # in debug in CTA R1 debug
            cs_serial=camera_config.debug.cs_serial,
            idaq_version=camera_config.debug.evb_version,
            cdhs_version=camera_config.debug.cdhs_version,
            tdp_type=camera_config.debug.tdp_type,
            tdp_action=camera_config.debug.tdp_action,
            ttype_pattern=camera_config.debug.ttype_pattern,
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

        lst_evt.pixel_status = np.zeros(N_PIXELS, dtype=zfits_event.pixel_status.dtype)
        lst_evt.pixel_status[self.camera_config.expected_pixels_id] = zfits_event.pixel_status

        # set bits for dvr if not already set
        if not self.dvr_applied:
            not_broken = get_channel_info(lst_evt.pixel_status) != 0
            lst_evt.pixel_status[not_broken] |= PixelStatus.DVR_STATUS_0

        lst_evt.ped_id = zfits_event.ped_id
        lst_evt.module_status = zfits_event.lstcam.module_status
        lst_evt.extdevices_presence = zfits_event.lstcam.extdevices_presence

        # if TIB data are there
        if lst_evt.extdevices_presence & 1:
            tib = zfits_event.lstcam.tib_data.view(TIB_DTYPE)[0]
            lst_evt.tib_event_counter = tib['event_counter']
            lst_evt.tib_pps_counter = tib['pps_counter']
            lst_evt.tib_tenMHz_counter = parse_tib_10MHz_counter(tib['tenMHz_counter'])
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

        if trigger.event_type == EventType.FLATFIELD:
            waveform = array_event.r1.tel[tel_id].waveform
            if waveform is not None and waveform.ndim == 2:
                self.log.warning(f'Event {array_event.index.event_id} tagged as FLATFIELD, but has only one gain!')

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
        
        if waveform.ndim == 3:
            image = waveform[HIGH_GAIN].sum(axis=1)
        else:
            image = waveform.sum(axis=1)

        in_range = (image >= self.min_flatfield_adc) & (image <= self.max_flatfield_adc)
        n_in_range = np.count_nonzero(in_range)

        looks_like_ff = n_in_range >= self.min_flatfield_pixel_fraction * image.size
        
        if looks_like_ff:
            # Tag as FF only events with 2-gains waveforms: both gains are needed for calibration
            if waveform.ndim == 3:
                array_event.trigger.event_type = EventType.FLATFIELD
                self.log.debug(
                    'Setting event type of event'
                    f' {array_event.index.event_id} to FLATFIELD'
                )
            else:
                array_event.trigger.event_type = EventType.UNKNOWN
                self.log.warning(
                    'Found FF-looking event that has just one gain:'
                    f'{array_event.index.event_id}. Setting event type to UNKNOWN'
                )
        elif array_event.trigger.event_type == EventType.FLATFIELD:
            self.log.warning(
                'Found FF event that does not fulfill FF criteria:'
                f'{array_event.index.event_id}. Setting event type to UNKNOWN' 
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
        n_pixels = self.n_pixels
        n_samples = self.n_samples
        pixel_id_map = self.pixel_id_map

        has_low_gain = (zfits_event.pixel_status & PixelStatus.LOW_GAIN_STORED).astype(bool)
        has_high_gain = (zfits_event.pixel_status & PixelStatus.HIGH_GAIN_STORED).astype(bool)
        broken_pixels = ~(has_low_gain | has_high_gain)

        # broken pixels have both false, so gain selected means checking
        # if there are any pixels where exactly one of high or low gain is stored
        gain_selected = np.any(has_low_gain != has_high_gain)

        # fill value for broken pixels
        dtype = zfits_event.waveform.dtype
        fill = np.iinfo(dtype).max

        if self.dvr_applied:
            stored_pixels = (zfits_event.pixel_status & PixelStatus.DVR_STATUS) > 0
        else:
            stored_pixels = slice(None)  # all pixels stored

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

            reordered_waveform = np.full((N_PIXELS, n_samples), fill, dtype=dtype)
            reordered_waveform[pixel_id_map[stored_pixels]] = waveform

            reordered_selected_gain = np.full(N_PIXELS, -1, dtype=np.int8)
            reordered_selected_gain[pixel_id_map] = selected_gain

            r0 = R0CameraContainer()
            r1 = R1CameraContainer(
                waveform=reordered_waveform,
                selected_gain_channel=reordered_selected_gain,
            )
        else:
            reshaped_waveform = zfits_event.waveform.reshape(N_GAINS, -1, n_samples)
            # re-order the waveform following the expected_pixels_id values
            reordered_waveform = np.full((N_GAINS, N_PIXELS, N_SAMPLES), fill, dtype=dtype)
            reordered_waveform[:, pixel_id_map[stored_pixels], :] = reshaped_waveform
            r0 = R0CameraContainer(waveform=reordered_waveform)
            r1 = R1CameraContainer()

        if CTAPIPE_0_20:
            # reorder to nominal pixel order
            pixel_status = _reorder_pixel_status(
                zfits_event.pixel_status, pixel_id_map, set_dvr_bits=True
            )
            r1.pixel_status = pixel_status
            r1.event_time = cta_high_res_to_time(
                zfits_event.trigger_time_s, zfits_event.trigger_time_qns,
            )

        return r0, r1

    def fill_r0r1_container(self, array_event, zfits_event):
        """
        Fill with R0Container

        """
        r0, r1 = self.fill_r0r1_camera_container(zfits_event)
        array_event.r0.tel[self.tel_id] = r0
        array_event.r1.tel[self.tel_id] = r1

    def initialize_mon_container(self):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        shape = (N_GAINS, N_PIXELS)
        # all pixels broken by default
        status_container = PixelStatusContainer(
            hardware_failing_pixels=np.ones(shape, dtype=bool),
            pedestal_failing_pixels=np.zeros(shape, dtype=bool),
            flatfield_failing_pixels=np.zeros(shape, dtype=bool),
        )

        camera_container = MonitoringCameraContainer(
            pixel_status = status_container,
        )
        container = MonitoringContainer()
        container.tel[self.tel_id] = camera_container
        return container

    def fill_mon_container(self, array_event, zfits_event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        status_container = array_event.mon.tel[self.tel_id].pixel_status

        # reorder the array
        pixel_id_map = self.pixel_id_map

        reordered_pixel_status = np.zeros(N_PIXELS, dtype=zfits_event.pixel_status.dtype)
        reordered_pixel_status[pixel_id_map] = zfits_event.pixel_status

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

