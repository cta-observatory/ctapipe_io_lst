from enum import IntEnum, IntFlag
from .constants import TriggerBits
from collections import defaultdict

class EVBPreprocessing(IntEnum):
    """
    The preprocessing steps that can be applied by EVB.

    The values of this Enum is the bit index of this step in the tdp_action value.

    Notes
    -----
    This was supposed to be documented in the EVB ICD:
    https://edms.cern.ch/ui/file/2411710/2.6/LSTMST-ICD-20191206.pdf
    But that document doest match the current EVB code.

    As of 2024-01-26, there is a bug in EVB that the ttype_pattern and
    tdp_action arrays are actually mixed up in the camera_configuration
    object.
    """
    # pre-processing flags
    GAIN_SELECTION = 0        # PPF0
    BASELINE_SUBTRACTION = 1  # PPF1
    DELTA_T_CORRECTION = 2    # PPF2
    SPIKE_REMOVAL = 3         # PPF3

    # processing flags
    PEDESTAL_SUBTRACTION = 5  # PF0
    PE_CALIBRATION = 6  # PF0


class EVBPreprocessingFlag(IntFlag):
    """
    IntFlag version of the EVBPreprocessing, as stored in Event.debug.tdp_action
    """
    GAIN_SELECTION = 1 << EVBPreprocessing.GAIN_SELECTION
    BASELINE_SUBTRACTION = 1 << EVBPreprocessing.BASELINE_SUBTRACTION
    DELTA_T_CORRECTION = 1 << EVBPreprocessing.DELTA_T_CORRECTION
    SPIKE_REMOVAL = 1 << EVBPreprocessing.SPIKE_REMOVAL

    PEDESTAL_SUBTRACTION = 1 << EVBPreprocessing.PEDESTAL_SUBTRACTION
    PE_CALIBRATION = 1 << EVBPreprocessing.PE_CALIBRATION


def get_processings_for_trigger_bits(camera_configuration):
    """
    Parse the tdp_action/type information into a dict mapping 
    """
    tdp_type = camera_configuration.debug.tdp_type

    # EVB has a bug, it stores the tdp_action in the wrong field
    tdp_action = camera_configuration.debug.ttype_pattern

    # first entry is default handling
    # only look at the first byte for now (maximum 6 bits defied above)
    default = EVBPreprocessingFlag(int(tdp_action[0]) & 0xff)
    actions = defaultdict(lambda: default)

    # the following entries refer to the entries in tdp_type
    # but with offset of 1, because 0 is the default
    for i, trigger_bits in enumerate(tdp_type, start=1): 
        # all-zero trigger bits can be ignored
        if trigger_bits == 0:
            continue

        # only look at the first byte for now (maximum 6 bits defied above)
        actions[TriggerBits(int(trigger_bits))] = EVBPreprocessingFlag(int(tdp_action[i]) & 0xff)

    return actions
