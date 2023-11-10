from enum import IntEnum
from .constants import TriggerBits
from collections import defaultdict

class EVBPreprocessing(IntEnum):
    """
    The preprocessing steps that can be applied by EVB.

    The values of this Enum is the index of this step in the tdp_action array.

    See EVB ICD section in
    https://edms.cern.ch/ui/file/2411710/2.6/LSTMST-ICD-20191206.pdf
    """
    GAIN_SELECTION = 0
    BASELINE_SUBTRACTION = 1
    DELTA_T_CORRECTION = 2
    KEEP_ALL_EVENTS = 3
    MUON_SEARCH = 4
    PEDESTAL_SUBTRACTION = 5
    CALIBRATION = 6
    PEDESTAL_SUM = 7
    CALIBRATION_SUM = 8


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
