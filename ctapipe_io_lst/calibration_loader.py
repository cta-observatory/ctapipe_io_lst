from typing import Protocol
from functools import lru_cache

import tables
import numpy as np

from ctapipe.containers import MonitoringContainer
from ctapipe.io import HDF5TableReader, read_table

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Path, TelescopeParameter
)

from .constants import (
    N_GAINS, N_PIXELS, N_MODULES, N_SAMPLES,
    N_PIXELS_MODULE, N_CAPACITORS_PIXEL, N_CAPACITORS_CHANNEL,
)


class CalibrationLoader(Protocol):

    def is_calibration_available(self):
        """ Tell if data can be calibrated (valid path to calibration data). """

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

    def load_drs4_spike_height(self, tel_id):
        """ Load the spike height from drs4 baseline data. """



class HDF5CalibrationLoader(TelescopeComponent):

    calibration_path = Path(
        None, exists=True, directory_ok=False, allow_none=True,
        help='Path to LST calibration file',
    ).tag(config=True)

    drs4_pedestal_path = TelescopeParameter(
        trait=Path(exists=True, directory_ok=False, allow_none=True),
        allow_none=True,
        default_value=None,
        help=(
            'Path to the LST pedestal file'
            ', required when `apply_drs4_pedestal_correction=True`'
            ' or when using spike subtraction'
        ),
    ).tag(config=True)

    drs4_time_calibration_path = TelescopeParameter(
        trait=Path(exists=True, directory_ok=False, allow_none=True),
        help='Path to the time calibration file',
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    def is_calibration_available(self):
        return self.calibration_path is not None

    def load_calibration_data(self):
        path = self.calibration_path
        if path is None:
            return None
        return self._load_calibration_data(path)

    def load_drs4_baseline_data(self, tel_id):
        path = self.drs4_pedestal_path.tel[tel_id]
        if path is None:
            raise ValueError(
                "DRS4 pedestal correction requested"
                " but no file provided for telescope"
            )

        return self._load_drs4_baseline_data(path, tel_id)

    def load_drs4_time_calibration_data(self, tel_id):
        path = self.drs4_time_calibration_path.tel[tel_id]
        if path is None:
            return None
        return self._load_drs4_time_calibration_data(path)

    def load_spike_heights(self, tel_id):
        path = self.drs4_pedestal_path.tel[tel_id]
        if path is None:
            raise ValueError(
                "DRS4 spike correction requested"
                " but no pedestal file provided for telescope"
            )

        return self._load_spike_heights(path, tel_id)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_drs4_baseline_data(path, tel_id):
        """
        Function to load pedestal file.

        To make boundary conditions unnecessary,
        the first N_SAMPLES values are repeated at the end of the array

        The result is cached so we can repeatedly call this method
        using the configured path without reading it each time.
        """
        table = read_table(path, f'/r1/monitoring/drs4_baseline/tel_{tel_id:03d}')

        pedestal_data = np.empty(
            (N_GAINS, N_PIXELS_MODULE * N_MODULES, N_CAPACITORS_PIXEL + N_SAMPLES),
            dtype=np.float32
        )
        pedestal_data[:, :, :N_CAPACITORS_PIXEL] = table[0]['baseline_mean']
        pedestal_data[:, :, N_CAPACITORS_PIXEL:] = pedestal_data[:, :, :N_SAMPLES]

        return pedestal_data

    @staticmethod
    def _load_calibration_data(path):
        """
        Read the correction from hdf5 calibration file
        """
        mon = MonitoringContainer()

        with tables.open_file(path) as f:
            tel_ids = [
                int(key[4:]) for key in f.root._v_children.keys()
                if key.startswith('tel_')
            ]

        for tel_id in tel_ids:
            with HDF5TableReader(path) as h5_table:
                base = f'/tel_{tel_id}'
                # read the calibration data
                table = base + '/calibration'
                next(h5_table.read(table, mon.tel[tel_id].calibration))

                # read pedestal data
                table = base + '/pedestal'
                next(h5_table.read(table, mon.tel[tel_id].pedestal))

                # read flat-field data
                table = base + '/flatfield'
                next(h5_table.read(table, mon.tel[tel_id].flatfield))

                # read the pixel_status container
                table = base + '/pixel_status'
                next(h5_table.read(table, mon.tel[tel_id].pixel_status))

        return mon

    @staticmethod
    def _load_drs4_time_calibration_data(path):
        """
        Function to load calibration file.
        """
        with tables.open_file(path, 'r') as f:
            fan = f.root.fan[:]
            fbn = f.root.fbn[:]

        return fan, fbn

    @lru_cache(maxsize=4)
    def _load_spike_heights(self, path, tel_id):
        table = read_table(path, f'/r1/monitoring/drs4_baseline/tel_{tel_id:03d}')
        spike_height = np.array(table[0]['spike_height'])
        return spike_height
