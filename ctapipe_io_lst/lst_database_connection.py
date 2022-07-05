import io
import zlib
import numpy as np
import psycopg2 as pg

def _adapt_array(text):
    """ Adapt a numpy array and return a Postgres binary object. """
    out = io.BytesIO()
    np.save(out, text)
    out.seek(0)
    return pg.Binary(zlib.compress(out.read()))

def _typecast_array(value, cur):
    """ Convert back a Postgres binary object into a numpy array. """
    if value is None:
        return None

    data = pg.BINARY(value, cur)
    bdata = io.BytesIO(zlib.decompress(data))
    bdata.seek(0)
    return np.load(bdata, allow_pickle=True)

pg.extensions.register_adapter(np.ndarray, _adapt_array)
t_array = pg.extensions.new_type(pg.BINARY.values, "numpy", _typecast_array)
pg.extensions.register_type(t_array)


def get_waveform_calibration_fields():
    """ Fields of the waveform calibration table.  """
    return [
        'time',
        'time_min',
        'time_max',
        'n_pe',
        'dc_to_pe',
        'time_correction',
        'pedestal_per_sample',
        'unusable_pixels',
    ]


def get_flatfield_calibration_fields():
    """ Fields of flatfield calibration table.  """
    return [
        'n_events',
        'sample_time',
        'sample_time_min',
        'sample_time_max',
        'charge_mean',
        'charge_median',
        'charge_std',
        'charge_median_outliers',
        'charge_std_outliers',
        'time_mean',
        'time_median',
        'time_std',
        'time_median_outliers',
        'relative_gain_mean',
        'relative_gain_median',
        'relative_gain_std',
        'relative_time_median',
    ]


def get_pedestal_calibration_fields():
    """ Fields of pedestal calibration table.  """
    return [
        'n_events',
        'sample_time',
        'sample_time_min',
        'sample_time_max',
        'charge_mean',
        'charge_median',
        'charge_std',
        'charge_median_outliers',
        'charge_std_outliers',
    ]


def get_pixel_status_fields():
    """ Fields of pixel status calibration table.  """
    return [
        'hardware_failing_pixels',
        'pedestal_failing_pixels',
        'flatfield_failing_pixels',
    ]


class LSTDatabaseConnection:

    def __init__(self, user, database, autocommit=False):
        """ Initialise the instance. The connection is not open.  """
        self.autocommit = autocommit
        self.user = user
        self.database = database

    def load_calibration_data(self, run):
        """
        Load the calibration data corresponding to a given run.
        (waveform, flatfield, pedestal and status)
        """
        waveform_fields = get_waveform_calibration_fields()
        self.cursor.execute(
            "SELECT " + ','.join(waveform_fields)
            + " FROM waveform_calibration_pro where run=%s",
            (run,))
        res_calib = self.cursor.fetchone()
        calib = {
            field_name: res_calib[i]
            for i, field_name in enumerate(waveform_fields)
        }

        flatfield_fields = get_flatfield_calibration_fields()
        self.cursor.execute(
            "SELECT " + ','.join(flatfield_fields)
            + " FROM flatfield_calibration_pro where run=%s",
            (run,))
        res_flatfield = self.cursor.fetchone()
        flatfield = {
            field_name: res_flatfield[i]
            for i, field_name in enumerate(flatfield_fields)
        }

        pedestal_fields = get_pedestal_calibration_fields()
        self.cursor.execute(
            "SELECT " + ','.join(pedestal_fields)
            + " FROM pedestal_calibration_pro where run=%s",
            (run,))
        res_pedestal = self.cursor.fetchone()
        pedestal = {
            field_name: res_pedestal[i]
            for i, field_name in enumerate(pedestal_fields)
        }

        pixel_status_fields = get_pixel_status_fields()
        self.cursor.execute(
            "SELECT " + ','.join(pixel_status_fields)
            + " FROM pixel_status_pro where run=%s",
            (run,))
        res_pixel = self.cursor.fetchone()
        pixel = {
            field_name: res_pixel[i]
            for i, field_name in enumerate(pixel_status_fields)
        }

        return {
            'calibration': calib,
            'flatfield': flatfield,
            'pedestal': pedestal,
            'pixel_status': pixel
        }

    def load_drs4_pedestal_data(self, run):
        """
        Load the drs4 pedestal data corresponding to a given run.
        """
        self.cursor.execute(
            "SELECT baseline_mean FROM drs4_baseline_pro where run=%s",
            (run,)
        )
        return self.cursor.fetchone()[0]

    def load_drs4_spike_height_data(self, run):
        """
        Load the drs4 spike data corresponding to a given run.
        """
        self.cursor.execute(
            "SELECT spike_height FROM drs4_baseline_pro where run=%s",
            (run,)
        )
        return self.cursor.fetchone()[0]

    def load_drs4_time_calibration_data(self, run):
        """
        Load the drs4 time calibration data corresponding to a given run.
        """
        self.cursor.execute(
            "SELECT fan, fbn FROM \"drs4_time_sampling_from_FF_pro\" where run=%s",
            (run,)
        )
        return self.cursor.fetchone()

    def open(self):
        """ Open the connection to the database. """
        print(f'Connection to db {self.database!r}, user is {self.user!r}')
        self.connection = pg.connect(f'dbname={self.database} user={self.user}')
        self.cursor = self.connection.cursor()
        return self

    def close(self):
        """ Close the connection to the database. """
        if self.connection and self.autocommit:
            self.connection.commit()
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def __enter__(self):
        """ Enter a context, open the connection. """
        self.open()
        return self

    def __exit__(self, *args):
        """ Exit a context, close the connection. """
        self.close()

