import numpy as np
from astropy.table import Table
from .constants import N_PIXELS_MODULE 

def get_pixel_table(pixel_id_map, module_id_map):
    """
    Construct a table of pixel / module ids from the mappings in CameraConfiguration
    """
    module_index = np.repeat(np.arange(len(module_id_map)), N_PIXELS_MODULE)
    pixel_index = np.arange(len(pixel_id_map))

    local_pixel_index = pixel_index % N_PIXELS_MODULE
    hardware_pixel_id = module_id_map * N_PIXELS_MODULE + local_pixel_index

    table = Table(dict(
        pixel_id=pixel_id_map,
        pixel_index=pixel_index,
        hardware_pixel_id=hardware_pixel_id,
        module_id=module_id_map,
        local_pixel_index=local_pixel_index,
        module_index=module_index,
    ))

    table.sort("pixel_id")
    return table
