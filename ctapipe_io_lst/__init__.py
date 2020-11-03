# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.
"""
import numpy as np
from astropy import units as u
from pkg_resources import resource_filename
import os
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

from ctapipe.io import EventSource
from ctapipe.io.datalevels import DataLevel
from ctapipe.core.traits import Int, Bool
from ctapipe.containers import PixelStatusContainer

from .containers import LSTArrayEventContainer
from .version import get_version
from .calibration import LSTR0Corrections


__version__ = get_version(pep440=False)
__all__ = ['LSTEventSource']


OPTICS = OpticsDescription(
    'LST',
    equivalent_focal_length=u.Quantity(28, u.m),
    num_mirrors=1,
    mirror_area=u.Quantity(386.73, u.m**2),
    num_mirror_tiles=198,
)


DRAGON_COUNTERS_DTYPE = np.dtype([
    ('pps_counter', np.uint16),
    ('tenMHz_counter', np.uint32),
    ('event_counter', np.uint32),
    ('trigger_counter', np.uint32),
    ('local_clock_counter', np.uint64),
]).newbyteorder('<')


TIB_DTYPE = np.dtype([
    ('event_counter', np.uint32),
    ('pps_counter', np.uint16),
    ('tenMHz_counter', np.uint32),
    ('stereo_pattern', np.uint8),
    ('masked_trigger', np.uint8),
]).newbyteorder('<')

CDTS_AFTER_37201_DTYPE = np.dtype([
    ('timestamp', np.uint64),
    ('address', np.float32),
    ('event_counter', np.float32),
    ('busy_counter', np.float32),
    ('pps_counter', np.float32),
    ('clock_counter', np.float32),
    ('trigger_type', np.uint8),
    ('white_rabbit_status', np.uint8),
    ('stereo_pattern', np.uint8),
    ('num_in_bunch', np.uint8),
    ('cdts_version', np.uint32),
]).newbyteorder('<')

CDTS_BEFORE_37201_DTYPE = np.dtype([
    ('event_counter', np.float32),
    ('pps_counter', np.float32),
    ('clock_counter', np.float32),
    ('timestamp', np.float64),
    ('camera_timestamp', np.float64),
    ('trigger_type', np.uint8),
    ('white_rabbit_status', np.uint8),
    ('unknown', np.uint8),
]).newbyteorder('<')

SWAT_DTYPE = np.dtype([
    ('timestamp', np.uint64),
    ('counter1', np.uint32),
    ('counter2', np.uint32),
    ('event_type', np.uint8),
    ('camera_flag', np.uint8),
    ('camera_event_num', np.uint32),
    ('array_flag', np.uint8),
    ('event_num', np.uint32),
]).newbyteorder('<')


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
        'resources/no_corrections_pulse_LST.dat'
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
        help='Number of gains at r0/r1 level'
    ).tag(config=True)

    baseline = Int(
        400,
        help='r0 waveform baseline (default from EvB v3)'
    ).tag(config=True)

    multi_streams = Bool(
        True,
        help='Read in parallel all streams '
    ).tag(config=True)

    def __init__(self, input_url=None, **kwargs):
        """
        Parameters
        ----------
        n_gains = number of gains expected in input file

        baseline = baseline to be subtracted at r1 level (not used for the moment)

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

    @property
    def subarray(self):
        return self._subarray

    @property
    def is_simulation(self):
        return False

    @property
    def obs_id(self):
        # currently no obs id is available from the input files
        return self.camera_config.configuration_id

    @property
    def datalevels(self):
        if self.r0_r1_calibrator.drs4_pedestal_path.tel[self.tel_id] is not None:
            return (DataLevel.R0, DataLevel.R1)
        return (DataLevel.R0)

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

        subarray = SubarrayDescription("LST1 subarray")
        subarray.tel_descriptions = tel_descriptions
        subarray.positions = tel_positions
        subarray.tel[tel_id] = lst_tel_descr

        return subarray

    def _generator(self):

        # container for LST data
        self.data = LSTArrayEventContainer()
        self.data.meta['input_url'] = self.input_url
        self.data.meta['max_events'] = self.max_events
        self.data.meta['origin'] = 'LSTCAM'

        # fill LST data from the CameraConfig table
        self.fill_lst_service_container_from_zfile()

        # initialize general monitoring container
        self.initialize_mon_container()

        # loop on events
        for count, event in enumerate(self.multi_file):

            self.data.count = count
            self.data.index.event_id = event.event_id
            self.data.index.obs_id = self.obs_id

            # fill specific LST event data
            self.fill_lst_event_container_from_zfile(event)

            # fill general monitoring data
            self.fill_mon_container_from_zfile(event)

            # fill general R0 data
            self.fill_r0_container_from_zfile(event)
            if self.r0_r1_calibrator.drs4_pedestal_path.tel[self.tel_id] is not None:
                self.r0_r1_calibrator.calibrate(self.data)

            yield self.data

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

    def fill_lst_service_container_from_zfile(self):
        """
        Fill LSTServiceContainer with specific LST service data data
        (from the CameraConfig table of zfit file)

        """

        svc_container = self.data.lst.tel[self.tel_id].svc

        svc_container.telescope_id = self.tel_id
        svc_container.cs_serial = self.camera_config.cs_serial
        svc_container.configuration_id = self.camera_config.configuration_id
        svc_container.date = self.camera_config.date
        svc_container.num_pixels = self.camera_config.num_pixels
        svc_container.num_samples = self.camera_config.num_samples
        svc_container.pixel_ids = self.camera_config.expected_pixels_id
        svc_container.data_model_version = self.camera_config.data_model_version
        svc_container.num_modules = self.camera_config.lstcam.num_modules
        svc_container.module_ids = self.camera_config.lstcam.expected_modules_id
        svc_container.idaq_version = self.camera_config.lstcam.idaq_version
        svc_container.cdhs_version = self.camera_config.lstcam.cdhs_version
        svc_container.algorithms = self.camera_config.lstcam.algorithms
        svc_container.pre_proc_algorithms = self.camera_config.lstcam.pre_proc_algorithms

    def fill_lst_event_container_from_zfile(self, event):
        """
        Fill LSTEventContainer with specific LST service data
        (from the Event table of zfit file)

        """

        event_container = self.data.lst.tel[self.tel_id].evt

        event_container.configuration_id = event.configuration_id
        event_container.event_id = event.event_id
        event_container.tel_event_id = event.tel_event_id
        event_container.pixel_status = event.pixel_status
        event_container.ped_id = event.ped_id
        event_container.module_status = event.lstcam.module_status
        event_container.extdevices_presence = event.lstcam.extdevices_presence

        # if TIB data are there
        if event_container.extdevices_presence & 1:
            tib = event.lstcam.tib_data.view(TIB_DTYPE)[0]
            event_container.tib_event_counter = tib['event_counter']
            event_container.tib_pps_counter = tib['pps_counter']
            event_container.tib_tenMHz_counter = tib['tenMHz_counter']
            event_container.tib_stereo_pattern = tib['stereo_pattern']
            event_container.tib_masked_trigger = tib['masked_trigger']

        # if UCTS data are there
        if event_container.extdevices_presence & 2:

            if int(self.data.lst.tel[self.tel_id].svc.idaq_version) > 37201:
                cdts = event.lstcam.cdts_data.view(CDTS_AFTER_37201_DTYPE)[0]
                event_container.ucts_timestamp = cdts[0]
                event_container.ucts_address = cdts[1]        # new
                event_container.ucts_event_counter = cdts[2]
                event_container.ucts_busy_counter = cdts[3]   # new
                event_container.ucts_pps_counter = cdts[4]
                event_container.ucts_clock_counter = cdts[5]
                event_container.ucts_trigger_type = cdts[6]
                event_container.ucts_white_rabbit_status = cdts[7]
                event_container.ucts_stereo_pattern = cdts[8] # new
                event_container.ucts_num_in_bunch = cdts[9]   # new
                event_container.ucts_cdts_version = cdts[10]  # new

            else:
                # unpack UCTS-CDTS data (old version)
                cdts = event.lstcam.cdts_data.view(CDTS_BEFORE_37201_DTYPE)[0]
                event_container.ucts_event_counter = cdts[0]
                event_container.ucts_pps_counter = cdts[1]
                event_container.ucts_clock_counter = cdts[2]
                event_container.ucts_timestamp = cdts[3]
                event_container.ucts_camera_timestamp = cdts[4]
                event_container.ucts_trigger_type = cdts[5]
                event_container.ucts_white_rabbit_status = cdts[6]

        # if SWAT data are there
        if event_container.extdevices_presence & 4:
            # unpack SWAT data
            unpacked_swat = event.lstcam.swat_data.view(SWAT_DTYPE)[0]
            event_container.swat_timestamp = unpacked_swat[0]
            event_container.swat_counter1 = unpacked_swat[1]
            event_container.swat_counter2 = unpacked_swat[2]
            event_container.swat_event_type = unpacked_swat[3]
            event_container.swat_camera_flag = unpacked_swat[4]
            event_container.swat_camera_event_num = unpacked_swat[5]
            event_container.swat_array_flag = unpacked_swat[6]
            event_container.swat_array_event_num = unpacked_swat[7]

        # unpack Dragon counters
        counters = event.lstcam.counters.view(DRAGON_COUNTERS_DTYPE)
        event_container.pps_counter = counters['pps_counter']
        event_container.tenMHz_counter = counters['tenMHz_counter']
        event_container.event_counter = counters['event_counter']
        event_container.trigger_counter = counters['trigger_counter']
        event_container.local_clock_counter = counters['local_clock_counter']

        event_container.chips_flags = event.lstcam.chips_flags
        event_container.first_capacitor_id = event.lstcam.first_capacitor_id
        event_container.drs_tag_status = event.lstcam.drs_tag_status
        event_container.drs_tag = event.lstcam.drs_tag

    def fill_trigger_info(self, trigger, event):
        module_rank = np.where(self.data.lst.tel[self.tel_id].svc.module_ids == 132)
        trigger.time = (
            self.data.lst.tel[self.tel_id].svc.date +
            self.data.lst.tel[self.tel_id].evt.pps_counter[module_rank] +
            self.data.lst.tel[self.tel_id].evt.tenMHz_counter[module_rank] * 10**(-7)
        )
        if self.data.lst.tel[self.tel_id].evt.tib_masked_trigger > 0:
            trigger.event_type = self.data.lst.tel[self.tel_id].evt.tib_masked_trigger
        else:
            trigger.event_type = -1

    def fill_r0_camera_container_from_zfile(self, r0_container, event):
        """
        Fill with R0CameraContainer
        """
        # verify the number of gains
        if event.waveform.shape[0] != self.camera_config.num_pixels * self.camera_config.num_samples * self.n_gains:
            raise ValueError(f"Number of gains not correct, waveform shape is {event.waveform.shape[0]}"
                             f" instead of "
                             f"{self.camera_config.num_pixels * self.camera_config.num_samples * self.n_gains}")

        reshaped_waveform = np.array(
            event.waveform
        ).reshape(
            self.n_gains,
            self.camera_config.num_pixels,
            self.camera_config.num_samples
        )

        # initialize the waveform container to zero
        n_camera_pixels = self.subarray.tel[self.tel_id].camera.geometry.n_pixels
        r0_container.waveform = np.zeros([self.n_gains, n_camera_pixels,
                                          self.camera_config.num_samples], dtype=np.float32)

        # re-order the waveform following the expected_pixels_id values
        # (rank = pixel id)
        r0_container.waveform[:, self.camera_config.expected_pixels_id, :] =\
            reshaped_waveform

    def fill_r0_container_from_zfile(self, event):
        """
        Fill with R0Container

        """
        container = self.data.r0

        r0_camera_container = container.tel[self.tel_id]
        self.fill_r0_camera_container_from_zfile(
            r0_camera_container,
            event
        )

    def initialize_mon_container(self):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        container = self.data.mon
        mon_camera_container = container.tel[self.tel_id]

        # initialize the container
        status_container = PixelStatusContainer()
        n_camera_pixels = self.subarray.tel[self.tel_id].camera.geometry.n_pixels
        status_container.hardware_failing_pixels = np.zeros((self.n_gains, n_camera_pixels), dtype=bool)
        status_container.pedestal_failing_pixels = np.zeros((self.n_gains, n_camera_pixels), dtype=bool)
        status_container.flatfield_failing_pixels = np.zeros((self.n_gains, n_camera_pixels), dtype=bool)

        mon_camera_container.pixel_status = status_container

    def fill_mon_container_from_zfile(self, event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """

        status_container = self.data.mon.tel[self.tel_id].pixel_status

        # reorder the array
        n_camera_pixels = self.subarray.tel[self.tel_id].camera.geometry.n_pixels
        pixel_status = np.zeros(n_camera_pixels)
        pixel_status[self.camera_config.expected_pixels_id] = event.pixel_status
        status_container.hardware_failing_pixels[:] = pixel_status == 0


class MultiFiles:

    """
    This class open all the files in file_list and read the events following
    the event_id order
    """

    def __init__(self, file_list):

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
                self._file[path] = File(path)
                self._events_table[path] = self._file[path].Events
                self._events[path] = next(self._file[path].Events)

                # verify where the CameraConfig is present
                if 'CameraConfig' in self._file[path].__dict__.keys():
                    self._camera_config[path] = next(self._file[path].CameraConfig)

                # for the moment it takes the first CameraConfig it finds (to be changed)
                    if(self.camera_config is None):
                        self.camera_config = self._camera_config[path]

            except StopIteration:
                pass

        # verify that somewhere the CameraConfing is present
        assert self.camera_config

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
