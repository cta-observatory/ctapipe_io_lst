import os
from pathlib import Path
import numpy as np
from astropy.time import Time
import astropy.units as u
from ctapipe.core import Provenance

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_drive_report = test_data / 'real/monitoring/DrivePositioning/DrivePosition_log_20200218.txt'
test_bending_report = test_data / 'real/monitoring/DrivePositioning/BendingModelCorrection_log_20220220.txt'
test_drive_report_with_bending = test_data / 'real/monitoring/DrivePositioning/DrivePosition_log_20220220.txt'
test_target_log = test_data / "real/monitoring/DrivePositioning/Target_log_20200218.txt"


def test_read_drive_report():
    from ctapipe_io_lst.pointing import PointingSource

    drive_report = PointingSource._read_drive_report(test_drive_report)

    assert 'time' not in drive_report.colnames
    assert 'azimuth' in drive_report.colnames
    assert 'zenith' in drive_report.colnames


def test_interpolation():
    from ctapipe_io_lst.pointing import PointingSource
    from ctapipe_io_lst import LSTEventSource

    subarray = LSTEventSource.create_subarray()
    pointing_source = PointingSource(
        subarray=subarray,
        drive_report_path=test_drive_report,
    )

    time = Time('2020-02-18T21:40:21')
    # relevent lines from drive log:
    # 1582062020 230.834 10.2514
    # 1582062022 230.896 10.2632
    pointing = pointing_source.get_pointing_position_altaz(tel_id=1, time=time)
    expected_alt = (90 - 0.5 * (10.2514 + 10.2632)) * u.deg
    assert u.isclose(pointing.altitude, expected_alt)
    assert u.isclose(pointing.azimuth, 0.5 * (230.834 + 230.896) * u.deg)

    # relevant lines from target log
    # 1582061035 TrackStart 86.6333 22.0144 OffCrabLo142 
    # 1582062466 TrackEnd 
    ra, dec = pointing_source.get_pointing_position_icrs(tel_id=1, time=time)
    assert u.isclose(ra, 86.6333 * u.deg)
    assert u.isclose(dec, 22.0144 * u.deg)


def test_bending_corrections():
    from ctapipe_io_lst.pointing import PointingSource
    corrections = PointingSource._read_bending_model_corrections(test_bending_report)
    assert corrections.colnames == ['unix_time', 'azimuth_correction', 'zenith_correction']


def test_load_position_and_bending_corrections():
    from ctapipe_io_lst.pointing import PointingSource

    Provenance().start_activity('test drive report')
    PointingSource._read_drive_report(test_drive_report_with_bending)
    inputs = Provenance().current_activity.input
    assert len(inputs) == 2
    assert inputs[0]['url'] == str(test_drive_report_with_bending.resolve())
    assert inputs[1]['url'] == str(test_bending_report.resolve())


def test_read_target_log(tmp_path):
    from ctapipe_io_lst.pointing import PointingSource

    targets = PointingSource._read_target_log(test_target_log)
    assert len(targets) == 7
    assert targets.colnames == ["start_unix", "ra", "dec", "name", "end_unix", "start", "end"]

    np.testing.assert_array_equal(targets["name"], ["Crab", "OffCrabLo142"] * 3 + ["Capella"])
    np.testing.assert_array_equal(targets["ra"], [83.6296, 86.6333] * 3 + [79.1725])
    np.testing.assert_array_equal(targets["dec"], [22.0144] * 6 + [45.9981])

    assert targets["ra"].unit == u.deg
    assert targets["dec"].unit == u.deg

    # test with empty file
    empty_log = (tmp_path / "Target_log.txt")
    empty_log.open("w").close()
    targets = PointingSource._read_target_log(empty_log)
    assert len(targets) == 0
    assert targets.colnames == ["start_unix", "ra", "dec", "name", "end_unix", "start", "end"]
    assert targets["ra"].unit == u.deg
    assert targets["dec"].unit == u.deg


def test_targets():
    from ctapipe_io_lst import PointingSource, LSTEventSource



    before_first_tracking = Time(1582058100, format="unix")
    crab = Time(1582062520, format="unix")
    between_obs = Time(1582066680, format="unix")
    capella = Time(1582069585, format="unix")
    after_last_tracking = Time(1582070175, format="unix")

    subarray = LSTEventSource.create_subarray(tel_id=1)

    # test explicitly giving path and with using drive_report_path
    test_kwargs = [
        dict(target_log_path=test_target_log),
        dict(drive_report_path=test_drive_report),
    ]

    for kwargs in test_kwargs:
        pointing = PointingSource(subarray, **kwargs)
        assert pointing.get_target(tel_id=1, time=before_first_tracking) is None
        assert pointing.get_target(tel_id=1, time=crab) == {
            "name": "Crab",
            "ra": u.Quantity(83.6296, u.deg),
            "dec": u.Quantity(22.0144, u.deg),
        }
        assert pointing.get_target(tel_id=1, time=capella) == {
            "name": "Capella",
            "ra": u.Quantity(79.1725, u.deg),
            "dec": u.Quantity(45.9981, u.deg),
        }
        assert pointing.get_target(tel_id=1, time=between_obs) is None
        assert pointing.get_target(tel_id=1, time=after_last_tracking) is None
