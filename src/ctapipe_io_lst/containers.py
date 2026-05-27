"""
Container structures for data that should be read or written to disk
"""
import numpy as np
from ctapipe.core import Container, Field, Map
from ctapipe.containers import ArrayEventContainer
from functools import partial
from astropy import units as u
from numpy import nan


__all__ = [
    'LSTEventContainer',
    'LSTServiceContainer',
    'LSTCameraContainer',
    'LSTContainer',
    'LSTArrayEventContainer',
    'PixelStatusContainer',
    'MonitoringCameraContainer',
]


class LSTServiceContainer(Container):
    """
    Container for Fields that are specific to each LST camera configuration
    """

    # Data from the CameraConfig table
    # present in CTA R1
    telescope_id = Field(-1, "telescope id")
    local_run_id = Field(-1, "local run id")
    date = Field(None, "config_time_s in data model")
    configuration_id = Field(None, "camera_config_id in the date model")
    pixel_ids = Field([], "pixel_id_map in the data model")
    module_ids = Field([], "module_id_map in the data model")

    num_modules = Field(-1, "number of modules")
    num_pixels = Field(-1, "number of pixels")
    num_channels = Field(-1, "number of channels")
    num_samples = Field(-1, "num samples")

    data_model_version = Field(None, "data model version")
    calibration_service_id = Field(-1, "calibration service id")
    calibration_algorithm_id = Field(-1, "calibration service id")

    # in debug in CTA R1 debug
    cs_serial = Field(None, "serial number of the camera server")
    idaq_version = Field(0, "idaq/evb version")
    cdhs_version = Field(0, "cdhs version")
    tdp_type = Field(None, "tdp type")
    tdp_action = Field(None, "tdp action")
    ttype_pattern = Field(None, "ttype_pattern")

    # only in old R1
    algorithms = Field(None, "algorithms")
    pre_proc_algorithms = Field(None, "pre processing algorithms")


class LSTEventContainer(Container):
    """
    Container for Fields that are specific to each LST event
    """
    # present in CTA R1 but not in ctapipe R1CameraEvent
    pixel_status = Field(None, "status of the pixels (n_pixels)", dtype=np.uint8)
    first_capacitor_id = Field(None, "first capacitor id")
    calibration_monitoring_id = Field(None, "calibration id of applied pre-calibration")
    local_clock_counter = Field(None, "Dragon local 133 MHz counter (n_modules)")

    # This one will eventually be present in R1CameraEvent:
    pixel_time_shift = Field(None, "Time shift in ns (n_channels, n_pixels)",
                             dtype=np.float32, ndim=2)


    # in debug event
    module_status = Field(None, "status of the modules (n_modules)")
    extdevices_presence = Field(None, "presence of data for external devices")
    chips_flags = Field(None, "chips flags")
    charges_hg = Field(None, "charges of high gain channel")
    charges_lg = Field(None, "charges of low gain channel")
    tdp_action = Field(None, "tdp action")

    tib_event_counter = Field(np.uint32(0), "TIB event counter", dtype=np.uint32)
    tib_pps_counter = Field(np.uint16(0), "TIB pps counter", dtype=np.uint16)
    tib_tenMHz_counter = Field(np.uint32(0), "TIB 10 MHz counter", dtype=np.uint32)
    tib_stereo_pattern = Field(np.uint16(0), "TIB stereo pattern", dtype=np.uint16)
    tib_masked_trigger = Field(0, "TIB trigger mask")

    ucts_event_counter =  Field(-1, "UCTS event counter")
    ucts_pps_counter = Field(-1, "UCTS pps counter")
    ucts_clock_counter = Field(-1, "UCTS clock counter")
    ucts_timestamp = Field(-1, "UCTS timestamp")
    ucts_camera_timestamp = Field(-1, "UCTS camera timestamp")
    ucts_trigger_type = Field(0, "UCTS trigger type")
    ucts_white_rabbit_status = Field(-1, "UCTS whiteRabbit status")
    ucts_address = Field(-1,"UCTS address")
    ucts_busy_counter = Field(-1, "UCTS busy counter")
    ucts_stereo_pattern = Field(0, "UCTS stereo pattern")
    ucts_num_in_bunch = Field(-1, "UCTS num in bunch (for debugging)")
    ucts_cdts_version = Field(-1, "UCTS CDTS version")

    swat_assigned_event_id = Field(np.uint64(0), "SWAT assigned event id")
    swat_event_request_bunch_id = Field(np.uint64(0), "SWAT event request bunch id")
    swat_trigger_request_id = Field(np.uint64(0), "SWAT trigger request bunch id")
    swat_trigger_id = Field(np.uint64(0), "SWAT trigger id")
    swat_bunch_id = Field(np.uint64(0), "SWAT bunch id")
    swat_trigger_type = Field(np.uint8(0), "SWAT trigger type")
    swat_trigger_time_s = Field(np.uint32(0), "SWAT trigger_time_s")
    swat_trigger_time_qns = Field(np.uint32(0), "SWAT trigger_time_qns")
    swat_readout_requested = Field(np.bool_(False), "SWAT readout requested")
    swat_data_available = Field(np.bool_(False), "SWAT data available")
    swat_hardware_stereo_trigger_mask = Field(np.uint16(0), "SWAT hardware stereo trigger mask")
    swat_negative_flag = Field(np.uint8(0), "SWAT negative flag")

    pps_counter= Field(None, "Dragon pulse per second counter (n_modules)")
    tenMHz_counter = Field(None, "Dragon 10 MHz counter (n_modules)")
    event_counter = Field(None, "Dragon event counter (n_modules)")
    trigger_counter = Field(None, "Dragon trigger counter (n_modules)")

    # Only in old R1
    configuration_id = Field(None, "id of the CameraConfiguration")
    event_id = Field(None, "global id of the event")
    tel_event_id = Field(None, "local id of the event")
    ped_id = Field(None, "tel_event_id of the event used for pedestal substraction")

    drs_tag_status = Field(None, "DRS tag status")
    drs_tag = Field(None, "DRS tag")

    # custom here
    ucts_jump = Field(False, "A ucts jump happened in the current event")


class LSTCameraContainer(Container):
    """
    Container for Fields that are specific to each LST camera
    """
    evt = Field(default_factory=LSTEventContainer, description="LST specific event Information")
    svc = Field(default_factory=LSTServiceContainer, description="LST specific camera_config Information")


class LSTContainer(Container):
    """
    Storage for the LSTCameraContainer for each telescope
    """

    # create the camera container
    tel = Field(
        default_factory=partial(Map, LSTCameraContainer),
        description="map of tel_id to LSTTelContainer"
    )


class PixelStatusContainer(Container):
    """
    Container for pixel status information

    It contains masks obtained by several data analysis steps
    At r0/r1 level only the hardware_mask is initialized
    """

    hardware_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the hardware pixel status data ("
        "n_chan, n_pix)",
    )

    pedestal_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the pedestal data analysis ("
        "n_chan, n_pix)",
    )

    flatfield_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the flat-field data analysis ("
        "n_chan, n_pix)",
    )


class FlatFieldContainer(Container):
    """
    Container for flat-field parameters obtained from a set of
    [n_events] flat-field events
    """

    sample_time = Field(
        0 * u.s, "Time associated to the flat-field event set ", unit=u.s
    )
    sample_time_min = Field(
        nan * u.s, "Minimum time of the flat-field events", unit=u.s
    )
    sample_time_max = Field(
        nan * u.s, "Maximum time of the flat-field events", unit=u.s
    )
    n_events = Field(0, "Number of events used for statistics")

    charge_mean = Field(None, "np array of signal charge mean (n_chan, n_pix)")
    charge_median = Field(None, "np array of signal charge median (n_chan, n_pix)")
    charge_std = Field(
        None, "np array of signal charge standard deviation (n_chan, n_pix)"
    )
    time_mean = Field(None, "np array of signal time mean (n_chan, n_pix)", unit=u.ns)
    time_median = Field(
        None, "np array of signal time median (n_chan, n_pix)", unit=u.ns
    )
    time_std = Field(
        None, "np array of signal time standard deviation (n_chan, n_pix)", unit=u.ns
    )
    relative_gain_mean = Field(
        None, "np array of the relative flat-field coefficient mean (n_chan, n_pix)"
    )
    relative_gain_median = Field(
        None, "np array of the relative flat-field coefficient  median (n_chan, n_pix)"
    )
    relative_gain_std = Field(
        None,
        "np array of the relative flat-field coefficient standard deviation (n_chan, n_pix)",
    )
    relative_time_median = Field(
        None,
        "np array of time (median) - time median averaged over camera (n_chan, n_pix)",
        unit=u.ns,
    )

    charge_median_outliers = Field(
        None, "Boolean np array of charge median outliers (n_chan, n_pix)"
    )
    charge_std_outliers = Field(
        None, "Boolean np array of charge std outliers (n_chan, n_pix)"
    )

    time_median_outliers = Field(
        None, "Boolean np array of pixel time (median) outliers (n_chan, n_pix)"
    )


class PedestalContainer(Container):
    """
    Container for pedestal parameters obtained from a set of
    [n_pedestal] pedestal events
    """

    n_events = Field(-1, "Number of events used for statistics")
    sample_time = Field(
        nan * u.s, "Time associated to the pedestal event set", unit=u.s
    )
    sample_time_min = Field(nan * u.s, "Time of first pedestal event", unit=u.s)
    sample_time_max = Field(nan * u.s, "Time of last pedestal event", unit=u.s)
    charge_mean = Field(None, "np array of pedestal average (n_chan, n_pix)")
    charge_median = Field(None, "np array of the pedestal  median (n_chan, n_pix)")
    charge_std = Field(
        None, "np array of the pedestal standard deviation (n_chan, n_pix)"
    )
    charge_median_outliers = Field(
        None, "Boolean np array of the pedestal median outliers (n_chan, n_pix)"
    )
    charge_std_outliers = Field(
        None, "Boolean np array of the pedestal std outliers (n_chan, n_pix)"
    )

class WaveformCalibrationContainer(Container):
    """
    Container for the pixel calibration coefficients
    """

    time = Field(nan * u.s, "Time associated to the calibration event", unit=u.s)
    time_min = Field(
        nan * u.s, "Earliest time of validity for the calibration event", unit=u.s
    )
    time_max = Field(
        nan * u.s, "Latest time of validity for the calibration event", unit=u.s
    )

    dc_to_pe = Field(
        None,
        "np array of (digital count) to (photon electron) coefficients (n_chan, n_pix)",
    )

    pedestal_per_sample = Field(
        None,
        "np array of average pedestal value per sample (digital count) (n_chan, n_pix)",
    )

    time_correction = Field(None, "np array of time correction values (n_chan, n_pix)")

    n_pe = Field(
        None, "np array of photo-electrons in calibration signal (n_chan, n_pix)"
    )

    unusable_pixels = Field(
        None,
        "Boolean np array of final calibration data analysis, True = failing pixels (n_chan, n_pix)",
    )

class MonitoringCameraContainer(Container):
    """
    Container for camera monitoring data
    """

    flatfield = Field(
        default_factory=FlatFieldContainer,
        description="Data from flat-field event distributions",
    )
    pedestal = Field(
        default_factory=PedestalContainer,
        description="Data from pedestal event distributions",
    )
    pixel_status = Field(
        default_factory=PixelStatusContainer,
        description="Container for masks with pixel status",
    )
    calibration = Field(
        default_factory=WaveformCalibrationContainer,
        description="Container for calibration coefficients",
    )

class MonitoringContainer(Container):
    """
    Root container for monitoring data (MON)
    """

    # create the camera container
    tel = Field(
        default_factory=partial(Map, MonitoringCameraContainer),
        description="map of tel_id to MonitoringCameraContainer",
    )


class LSTArrayEventContainer(ArrayEventContainer):
    """
    Data container including LST and monitoring information
    """
    lst = Field(default_factory=LSTContainer, description="LST specific Information")
    # Patch
    mon = Field(
        default_factory=MonitoringContainer,
        description="container for event-wise monitoring data (MON)",
    )