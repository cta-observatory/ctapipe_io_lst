import os
from pathlib import Path
from protozfits import File
from ctapipe_io_lst import TriggerBits
from ctapipe_io_lst.evb_preprocessing import EVBPreprocessingFlag

import pytest


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()

test_files = [
    "20231214/LST-1.1.Run16102.0000_first50.fits.fz",  # has only baseline enabled
    "20231218/LST-1.1.Run16231.0000_first50.fits.fz",  # baseline + gain selection for physics
    "20231219/LST-1.1.Run16255.0000_first50.fits.fz",  # all corrections + gain selection for physics
]

all_corrections = EVBPreprocessingFlag.BASELINE_SUBTRACTION | EVBPreprocessingFlag.DELTA_T_CORRECTION | EVBPreprocessingFlag.PE_CALIBRATION | EVBPreprocessingFlag.PEDESTAL_SUBTRACTION

expected = [
    {
        TriggerBits.MONO: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.STEREO: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.MONO|TriggerBits.PEDESTAL: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.MONO|TriggerBits.CALIBRATION: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.PEDESTAL: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.CALIBRATION: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
    },
    {
        TriggerBits.MONO: EVBPreprocessingFlag.BASELINE_SUBTRACTION | EVBPreprocessingFlag.GAIN_SELECTION,
        TriggerBits.STEREO: EVBPreprocessingFlag.BASELINE_SUBTRACTION | EVBPreprocessingFlag.GAIN_SELECTION,
        TriggerBits.MONO|TriggerBits.PEDESTAL: EVBPreprocessingFlag.BASELINE_SUBTRACTION | EVBPreprocessingFlag.GAIN_SELECTION,
        TriggerBits.MONO|TriggerBits.CALIBRATION: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.PEDESTAL: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
        TriggerBits.CALIBRATION: EVBPreprocessingFlag.BASELINE_SUBTRACTION,
    },
    {
        TriggerBits.MONO: all_corrections | EVBPreprocessingFlag.GAIN_SELECTION,
        TriggerBits.STEREO: all_corrections| EVBPreprocessingFlag.GAIN_SELECTION,
        TriggerBits.MONO|TriggerBits.PEDESTAL: all_corrections | EVBPreprocessingFlag.GAIN_SELECTION,
        TriggerBits.MONO|TriggerBits.CALIBRATION: all_corrections,
        TriggerBits.PEDESTAL: all_corrections,
        TriggerBits.CALIBRATION: all_corrections,
    },
]
@pytest.mark.parametrize(("test_file", "expected"), zip(test_files, expected))
def test_get_processings_for_trigger_bits(test_file, expected):
    from ctapipe_io_lst.evb_preprocessing import get_processings_for_trigger_bits

    path = test_data / "real/R0/" / test_file

    with File(str(path)) as f:
        camera_config = f.CameraConfiguration[0]

    result = get_processings_for_trigger_bits(camera_config)
    assert result == expected
