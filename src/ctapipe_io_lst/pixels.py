import numpy as np
from functools import lru_cache
from astropy.table import Table
from importlib.resources import files, as_file

from ctapipe.core import Provenance

from .constants import N_PIXELS_MODULE, CHANNEL_ORDER_LOW_GAIN, CHANNEL_ORDER_HIGH_GAIN, N_CHANNELS_MODULE


@lru_cache()
def pixel_channel_indices(n_modules):
    """Get index into first_capacitor_id or each channel, pixel"""
    return _pixel_channel_indices(np.arange(n_modules))


def _pixel_channel_indices(module_id_map):
    module = np.repeat(module_id_map, N_PIXELS_MODULE)
    n_modules = len(module_id_map)
    low_gain = module * N_CHANNELS_MODULE + np.tile(CHANNEL_ORDER_LOW_GAIN, n_modules)
    high_gain = module * N_CHANNELS_MODULE + np.tile(CHANNEL_ORDER_HIGH_GAIN, n_modules)
    return low_gain, high_gain


def get_pixel_table_from_camera_config(config):
    """
    Construct a table of pixel / module ids from the mappings in CameraConfiguration
    """
    if hasattr(config, "pixel_id_map"):
        # new R1v1.CameraConfiguration
        pixel_id_map = config.pixel_id_map
        module_id_map = config.module_id_map
    else:
        # old ProtoR1.CameraConfiguration
        pixel_id_map = config.expected_pixels_id
        module_id_map = config.lstcam.expected_modules_id

    n_modules = len(module_id_map)
    n_pixels = len(pixel_id_map)

    chip_lg, chip_hg = _pixel_channel_indices(module_id_map)

    pixel_index = np.arange(n_pixels)
    # the pixel data arrives in module groups, the module id of each group
    # is in module_id_map. Repeat to have the module_id of each pixel
    module_id = np.repeat(module_id_map, N_PIXELS_MODULE)

    # pixels inside one module are ordered by module_pixel_index
    module_pixel_index = np.tile(np.arange(N_PIXELS_MODULE), n_modules)
    hardware_pixel_id = module_id * N_PIXELS_MODULE + module_pixel_index

    table = Table(dict(
        pixel_id=pixel_id_map,
        pixel_index=pixel_index,
        module_id=module_id,
        module_pixel_index=module_pixel_index,
        hardware_pixel_id=hardware_pixel_id,
        drs4_chip_hg=chip_hg,
        drs4_chip_lg=chip_lg,
    ))

    table.sort("pixel_id")
    return table


def get_pixel_table():
    """Load pixel information from bundled camera geometry file"""
    with as_file(files("ctapipe_io_lst") / "resources/LSTCam.camgeom.fits.gz") as path:
        Provenance().add_input_file(path, role="CameraGeometry")
        return Table.read(path)
