from typing import Protocol

from ctapipe.containers import MonitoringContainer
from ctapipe.io import HDF5TableReader


class CalibrationLoader(Protocol):
    def load_calibration_data(self):
        """ Load the waveform calibration data. """

    def load_drs4_baseline_data(self):
        """ Load the drs4 baseline calibration data. """

    def load_drs4_time_calibration_data(self):
        """ Load drs4 time calibration from FF."""

    def load_drs4_time_calibration_data_for_tel(self, tel_id):
        """
        Load the drs4 time calibration from FF
        for a given telescope id.
        """
