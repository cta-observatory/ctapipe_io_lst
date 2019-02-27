"""
Container structures for data that should be read or written to disk
"""
from ctapipe.core import Container, Field, Map
from ctapipe.io.containers import DataContainer


__all__ = [
    'LSTContainer',
    'LSTCameraContainer',
    'LSTServiceContainer',
    'LSTEventContainer',
]


class LSTServiceContainer(Container):
    """
    Container for Fields that are specific to each LST camera configuration
    """

    # Data from the CameraConfig table
    telescope_id = Field(-1, "telescope id")
    cs_serial = Field(None, "serial number of the camera server")
    configuration_id = Field(None, "id of the CameraConfiguration")
    date = Field(None, "NTP start of run date")
    num_pixels = Field(-1, "number of pixels")
    num_samples = Field(-1, "num samples")
    pixel_ids = Field([], "id of the pixels in the waveform array")
    data_model_version = Field(None, "data model version")

    idaq_version = Field(0o0, "idaq version")
    cdhs_version = Field(0o0, "cdhs version")
    algorithms = Field(None, "algorithms")
    pre_proc_algorithms = Field(None, "pre processing algorithms")
    module_ids = Field([], "module ids")
    num_modules = Field(-1, "number of modules")


class LSTEventContainer(Container):
    """
    Container for Fields that are specific to each LST event
    """

    # Data from the CameraEvent table
    configuration_id = Field(None, "id of the CameraConfiguration")
    event_id = Field(None, "local id of the event")
    tel_event_id = Field(None, "global id of the event")
    pixel_status = Field([], "status of the pixels")
    ped_id = Field(None, "tel_event_id of the event used for pedestal substraction")
    module_status = Field([], "status of the modules")
    extdevices_presence = Field(None, "presence of data for external devices")
    tib_data = Field([], "TIB data array")
    cdts_data = Field([], "CDTS data array")
    swat_data = Field([], "SWAT data array")
    counters = Field([], "counters")
    chips_flags = Field([], "chips flags")
    first_capacitor_id = Field([], "first capacitor id")
    drs_tag_status = Field([], "DRS tag status")
    drs_tag = Field([], "DRS tag")


class LSTCameraContainer(Container):
    """
    Container for Fields that are specific to each LST camera
    """
    evt = Field(LSTEventContainer(), "LST specific event Information")
    svc = Field(LSTServiceContainer(), "LST specific camera_config Information")


class LSTContainer(Container):
    """
    Storage for the LSTCameraContainer for each telescope
    """
    tels_with_data = Field([], "list of telescopes with data")

    # create the camera container
    tel = Field(
        Map(LSTCameraContainer),
        "map of tel_id to LSTTelContainer")


class LSTDataContainer(DataContainer):
    """
    Data container including LST information
    """
    lst = Field(LSTContainer(), "LST specific Information")
