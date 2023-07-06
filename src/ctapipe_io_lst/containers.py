"""
Container structures for data that should be read or written to disk
"""
import numpy as np
from ctapipe.core import Container, Field, Map
from ctapipe.containers import ArrayEventContainer
from functools import partial


__all__ = [
    'LSTEventContainer',
    'LSTServiceContainer',
    'LSTCameraContainer',
    'LSTContainer',
    'LSTArrayEventContainer',
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

    swat_assigned_event_id = Field(np.uint32(0), "SWAT assigned event id")
    swat_trigger_id = Field(np.uint64(0), "SWAT trigger id")
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


class LSTArrayEventContainer(ArrayEventContainer):
    """
    Data container including LST and monitoring information
    """
    lst = Field(default_factory=LSTContainer, description="LST specific Information")
