from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from astropy.table import Table, Column
from astropy import units as u

from ctapipe.core import TelescopeComponent, Provenance
from ctapipe.core import traits
from ctapipe.containers import TelescopePointingContainer

from astropy.time import Time


__all__ = [
    'PointingSource'
]


NAN_ANGLE = np.nan * u.deg


class PointingSource(TelescopeComponent):
    """Provides access to pointing information stored in LST drive reports.
    """
    drive_report_path = traits.TelescopeParameter(
        trait=traits.Path(exists=True, directory_ok=False, allow_none=True),
        help=(
            'Path to the LST drive report file.'
            ' This has to be the new format, where files are called'
            ' DrivePosition_log_YYYYMMDD.txt'
            ' If a corresponding file BendingModelCorrection_log_YYYYMMDD.txt'
            ' is available next to the given path, it is read and used as well'
        ),
        default_value=None,
    ).tag(config=True)

    bending_model_corrections_path = traits.TelescopeParameter(
        trait=traits.Path(exists=True, directory_ok=False, allow_none=True),
        help=(
            'Path to the LST bending model corrections file.'
            ' This has to be the new format, where files are called'
            ' BendingModelCorrections_log_YYYYMMDD.txt'
            ' If this is None, but a drive position log is given, the PointingSource will look for '
            ' the corresponding BendingModelCorrection_log in the same directory'
        ),
        default_value=None,
    ).tag(config=True)

    target_log_path = traits.TelescopeParameter(
        trait=traits.Path(exists=True, directory_ok=False, allow_none=True),
        help=(
            'Path to the LST target log.'
            ' This has to be the file usually called Target_log_YYYYMMDD.txt'
            ' If this is None, but a drive position log is given, the PointingSource will look for '
            ' the corresponding Target_log in the same directory'
        ),
        default_value=None,
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        '''Initialize PointingSource'''

        super().__init__(subarray, config=config, parent=parent, **kwargs)
        self.drive_log = {}
        self.target_log = {}
        self.interp_az = {}
        self.interp_alt = {}

    @staticmethod
    def _read_target_log(path):
        path = Path(path)

        def parse_start(tokens):
            return {
                "start_unix": int(tokens[0]),
                "ra": float(tokens[2]),
                "dec": float(tokens[3]),
                "name": tokens[4],
            }

        def parse_end(tokens):
            return {"end_unix": int(tokens[0])}

        # state machine for Tracking/not Tracking
        Provenance().add_input_file(str(path), "target log")
        tracking = False
        targets = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                tokens = line.strip().split(" ")
                if tokens[1] == "TrackStart":
                    if tracking:
                        raise ValueError(f"Expected TrackingEnd, got {line}")
                    tracking = True
                    targets.append(parse_start(tokens))

                elif tokens[1] == "TrackEnd":
                    if not tracking:
                        raise ValueError(f"Expected TrackingStart, got {line}")
                    tracking = False
                    targets[-1].update(parse_end(tokens))

        if len(targets) > 0:
            table = Table(targets, units={"ra": u.deg, "dec": u.deg})
        else:
            table = Table([
                Column([], name="start_unix", dtype=int),
                Column([], name="ra", dtype=float, unit=u.deg),
                Column([], name="dec", dtype=float, unit=u.deg),
                Column([], name="name", dtype=str),
                Column([], name="end_unix", dtype=int),
            ])

        for col in ("start", "end"):
            table[col] = Time(table[f"{col}_unix"], format="unix")
            table[col].format = "isot"

        return table

    @staticmethod
    def _read_drive_report(path, bending_model_corrections_path=None):
        """
        Read a drive report into an astropy table

        Parameters
        ----------
        path : str or Path
            drive report file

        Returns
        -------
        data : `~astropy.table.Table`
            A table of drive reports
        """
        path = Path(path)
        Provenance().add_input_file(str(path), 'drive positioning')

        try:
            data = Table.read(
                path, format='ascii', delimiter=' ',
                header_start=None,
                data_start=0,
                names=["unix_time", "azimuth", "zenith"],
            )
        except Exception as e:
            raise IOError("Error reading drive report path") from e

        # check for bending model corrections
        # if the filename is not mathcing the general scheme, we cannot look for the bending model
        if bending_model_corrections_path is None:
            if "DrivePosition" not in path.name:
                return data

            bending_name = path.name.replace('DrivePosition', 'BendingModelCorrection')
            bending_model_corrections_path = path.with_name(bending_name)
            if not bending_model_corrections_path.exists():
                return data

        corrections = PointingSource._read_bending_model_corrections(bending_model_corrections_path)

        # according to an email by Armand Fiasson, the timestamps are guaranteed to be equal
        # but it might happen that one report has more rows than the other due to different
        # times when they are synced to fefs during the night
        if len(corrections) != len(data):
            n_common = min(len(corrections), len(data))
            corrections = corrections[:n_common]
            data = data[:n_common]

        if np.any(data['unix_time'] != corrections['unix_time']):
            raise IOError('Drive report and corrections timestamps differ')

        for col in ['azimuth', 'zenith']:
            data[col] += corrections[f'{col}_correction']

        return data

    @staticmethod
    def _read_bending_model_corrections(path):
        '''
        Read a bendingmodelcorrection report.
        '''
        Provenance().add_input_file(str(path), 'bending model corrections')
        return Table.read(
            path,
            format="ascii",
            delimiter=" ",
            header_start=None,
            data_start=0,
            names=["unix_time", "azimuth_correction", "zenith_correction"],
        )

    def _read_drive_report_for_tel(self, tel_id):
        path = self.drive_report_path.tel[tel_id]
        if path is None:
            raise ValueError(f'No drive report given for telescope {tel_id}')

        bending_model_corrections_path = self.bending_model_corrections_path.tel[tel_id]

        self.log.info(f'Loading drive report "{path}" for tel_id={tel_id}')
        self.drive_log[tel_id] = self._read_drive_report(path, bending_model_corrections_path)

        self.interp_az[tel_id] = interp1d(
            self.drive_log[tel_id]['unix_time'],
            self.drive_log[tel_id]['azimuth'],
        )
        self.interp_alt[tel_id] = interp1d(
            self.drive_log[tel_id]['unix_time'],
            90 - self.drive_log[tel_id]['zenith'],
        )

    def get_pointing_position_altaz(self, tel_id, time):
        """
        Calculating pointing positions by interpolation

        Parameters:
        -----------
        time: array
            times from events

        Drivereport: Container
            a container filled with drive information
        """
        if tel_id not in self.drive_log:
            self._read_drive_report_for_tel(tel_id)

        alt = u.Quantity(self.interp_alt[tel_id](time.unix), u.deg)
        az = u.Quantity(self.interp_az[tel_id](time.unix), u.deg)

        return TelescopePointingContainer(
            altitude=alt.to(u.rad),
            azimuth=az.to(u.rad),
        )

    def get_pointing_position_icrs(self, tel_id, time):
        """Return the target pointing position in ra/dec"""
        target = self.get_target(tel_id, time)
        if target is None:
            return NAN_ANGLE, NAN_ANGLE

        return target["ra"], target["dec"]

    def _get_target_log_path(self, tel_id):
        """Get the path for the Target_log_YYYYMMDD.txt file

        If the explicit traitlet is None, use the ``drive_report_path`` to
        look for a Target log in the same directory.
        """
        path = self.target_log_path.tel[tel_id]

        if path is not None:
            return path

        drive_path = self.drive_report_path.tel[tel_id]
        if drive_path is None:
            return None

        if "DrivePosition" in drive_path.name:
            target_name = drive_path.name.replace('DrivePosition', 'Target')
            target_path = drive_path.with_name(target_name)
            if target_path.exists():
                return target_path

        return None

    def get_target(self, tel_id, time):
        if tel_id not in self.target_log:
            path = self._get_target_log_path(tel_id)
            if path is None:
                self.target_log[tel_id] = None
            else:
                self.target_log[tel_id] = self._read_target_log(path)

        targets = self.target_log[tel_id]
        if targets is None:
            return

        time_unix = time.unix

        idx = np.searchsorted(targets["start_unix"], time_unix)

        # completely outside the available trackings
        if idx == 0 or idx > len(targets):
            return None

        row = targets[idx - 1]

        # start_unix <= time_unix is guaranteed by searchsorted
        if time_unix > row["end_unix"]:
            return None

        return {
            "name": row["name"],
            "ra": u.Quantity(row["ra"], targets["ra"].unit),
            "dec": u.Quantity(row["dec"], targets["dec"].unit)
        }
