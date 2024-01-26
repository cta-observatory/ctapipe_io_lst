from enum import IntEnum
from .constants import TriggerBits
from collections import defaultdict

class EVBPreprocessing(IntEnum):
    """
    The preprocessing steps that can be applied by EVB.

    The values of this Enum is the index of this step in the tdp_action array.

    Note
    ----
    This was supposed to be documented in the EVB ICD:
    https://edms.cern.ch/ui/file/2411710/2.6/LSTMST-ICD-20191206.pdf
    But that document doest match the current EVB code.
    """
    # pre-processing flags
    GAIN_SELECTION = 0        # PPF0
    BASELINE_SUBTRACTION = 1  # PPF1
    DELTA_T_CORRECTION = 2    # PPF2
    SPIKE_REMOVAL = 3         # PPF3
    RESERVED1 = 4             # PPF4

    # processing flags
    PEDESTAL_SUBTRACTION = 5  # PF0
    PE_CALIBRATION = 6  # PF0
    RESERVED2 = 7  # PF0
    RESERVED3 = 8  # PF0


def get_processings_for_trigger_bits(camera_configuration):
    """
    Parse the tdp_action/type information into a dict mapping 
    """
    tdp_type = camera_configuration.debug.tdp_type
    tdp_action = camera_configuration.debug.tdp_action

    # first bit (no shift) is default handling
    default = {step for step in EVBPreprocessing if tdp_action[step] & 1}
    actions = defaultdict(lambda: default)

    # the following bits refer to the entries in tdp_type
    for i, trigger_bits in enumerate(tdp_type, start=1): 
        # all-zero trigger bits can be ignored
        if trigger_bits == 0:
            continue

        actions[TriggerBits(int(trigger_bits))] = {
            step for step in EVBPreprocessing
            if tdp_action[step] & (1 << i)
        }

    return actions
