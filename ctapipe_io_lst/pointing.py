from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from astropy.table import Table
from astropy import units as u

from ctapipe.core import TelescopeComponent, Provenance
from ctapipe.core import traits
from ctapipe.containers import TelescopePointingContainer


__all__ = [
    'PointingSource'
]


NAN_ANGLE = np.nan * u.deg


class PointingSource(TelescopeComponent):
    """Provides access to pointing information stored in LST drive reports."""
    drive_report_path = traits.TelescopeParameter(
        trait=traits.Path(exists=True, directory_ok=False, allow_none=True),
        help='Path to the LST drive report file',
        default_value=None,
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        '''Initialize PointingSource'''

        super().__init__(subarray, config=config, parent=parent, **kwargs)
        self.drive_report = {}
        self.interp_az = {}
        self.interp_alt = {}
        self.interp_ra = {}
        self.interp_dec = {}

    @staticmethod
    def _read_drive_report(path):
        """
        Read a drive report into an astropy table

        Parameters:
        -----------
        str: drive report file

        Returns:
        data:`~astropy.table.Table`
             A table of drive reports

        """
        path = Path(path)
        Provenance().add_input_file(str(path), 'drive positioning')
        data = Table.read(
            path, format='ascii', delimiter=' ',
            header_start=None,
            data_start=0,
            # the Az, Zd, Ra, Dec columns are just dummies, the file
            # always contains "Az", "Zd" (or "El" for older files), "Ra", "Dec"
            names=[
                'weekday', 'month', 'day', 'time', 'year', 'unix_time',
                'Az', 'azimuth_avg', 'azimuth_min', 'azimuth_max', 'azimuth_std',
                'Zd', 'zenith_avg', 'zenith_min', 'zenith_max', 'zenith_std',
                'Ra', 'target_ra', 'Dec', 'target_dec',
            ],
            exclude_names=['weekday', 'month', 'day', 'time', 'year', 'Az', 'Zd', 'Ra', 'Dec'],
        )

        # check for bending model corrections
        bending_path = path.with_name(path.name.replace('drive', 'bendingmodelcorrection'))
        if not bending_path.exists():
            return data

        corrections = PointingSource._read_bending_model_corrections(bending_path)

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
            for key in ['avg', 'min', 'max']:
                data[f'{col}_{key}'] += corrections[f'{col}_correction']

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
            names=[
                "weekday",
                "month",
                "day",
                "time",
                "year",
                "unix_time",
                "Az", # dummy for the column always containing "Az"
                "azimuth_correction",
                # dummy for the column always containing "Zd" or "El" for older files
                "Zd",
                "zenith_correction",
            ],
            exclude_names=["weekday", "month", "day", "time", "year", "Az", "Zd"],
        )

    def _read_drive_report_for_tel(self, tel_id):
        path = self.drive_report_path.tel[tel_id]
        if path is None:
            raise ValueError(f'No drive report given for telescope {tel_id}')

        self.log.info(f'Loading drive report "{path}" for tel_id={tel_id}')
        self.drive_report[tel_id] = self._read_drive_report(path)

        self.interp_az[tel_id] = interp1d(
            self.drive_report[tel_id]['unix_time'],
            self.drive_report[tel_id]['azimuth_avg'],
        )
        self.interp_alt[tel_id] = interp1d(
            self.drive_report[tel_id]['unix_time'],
            90 - self.drive_report[tel_id]['zenith_avg'],
        )

        self.interp_ra[tel_id] = interp1d(
            self.drive_report[tel_id]['unix_time'],
            self.drive_report[tel_id]['target_ra'],
        )
        self.interp_dec[tel_id] = interp1d(
            self.drive_report[tel_id]['unix_time'],
            self.drive_report[tel_id]['target_dec'],
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
        if tel_id not in self.drive_report:
            self._read_drive_report_for_tel(tel_id)

        alt = u.Quantity(self.interp_alt[tel_id](time.unix), u.deg)
        az = u.Quantity(self.interp_az[tel_id](time.unix), u.deg)

        return TelescopePointingContainer(
            altitude=alt.to(u.rad),
            azimuth=az.to(u.rad),
        )

    def get_pointing_position_icrs(self, tel_id, time):
        if tel_id not in self.drive_report:
            self._read_drive_report_for_tel(tel_id)

        ra = u.Quantity(self.interp_ra[tel_id](time.unix), u.deg)
        dec = u.Quantity(self.interp_dec[tel_id](time.unix), u.deg)

        # drive reports contain 0 / 0 if not tracking ICRS coordinates
        # TODO: hope we never really observe ra=0°, dec=0°
        if ra != 0.0 and dec != 0.0:
            return ra, dec

        return NAN_ANGLE, NAN_ANGLE
