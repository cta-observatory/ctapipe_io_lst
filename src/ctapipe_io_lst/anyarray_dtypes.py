'''
Numpy dtypes for the structured arrays send as anyarray of
opaque bytes by EVB in LST R1 and CTA R1v1 debug events.

These data structures are defined in the EVB ICD:
https://edms.cern.ch/ui/file/2411710/2.6/LSTMST-ICD-20191206.pdf
'''
import numpy as np


DRAGON_COUNTERS_DTYPE = np.dtype([
    ('pps_counter', np.uint16),
    ('tenMHz_counter', np.uint32),
    ('event_counter', np.uint32),
    ('trigger_counter', np.uint32),
    ('local_clock_counter', np.uint64),
]).newbyteorder('<')


TIB_DTYPE = np.dtype([
    ('event_counter', np.uint32),
    ('pps_counter', np.uint16),
    ('tenMHz_counter', (np.uint8, 3)),
    ('stereo_pattern', np.uint16),
    ('masked_trigger', np.uint8),
]).newbyteorder('<')

CDTS_AFTER_37201_DTYPE = np.dtype([
    ('timestamp', np.uint64),
    ('address', np.uint32),
    ('event_counter', np.uint32),
    ('busy_counter', np.uint32),
    ('pps_counter', np.uint32),
    ('clock_counter', np.uint32),
    ('trigger_type', np.uint8),
    ('white_rabbit_status', np.uint8),
    ('stereo_pattern', np.uint8),
    ('num_in_bunch', np.uint8),
    ('cdts_version', np.uint32),
]).newbyteorder('<')

CDTS_BEFORE_37201_DTYPE = np.dtype([
    ('event_counter', np.uint32),
    ('pps_counter', np.uint32),
    ('clock_counter', np.uint32),
    ('timestamp', np.uint64),
    ('camera_timestamp', np.uint64),
    ('trigger_type', np.uint8),
    ('white_rabbit_status', np.uint8),
    ('unknown', np.uint8),  # called arbitraryInformation in C-Struct
]).newbyteorder('<')


SWAT_DTYPE = np.dtype([
    ("assigned_event_id", np.uint32),
    ("trigger_id", np.uint64),
    ("trigger_type", np.uint8),
    ("trigger_time_s", np.uint32),
    ("trigger_time_qns", np.uint32),
    ("readout_requested", np.bool_),
    ("data_available", np.bool_),
    ("hardware_stereo_trigger_mask", np.uint16),
    ("negative_flag", np.uint8),
], align=True).newbyteorder('<')


def parse_tib_10MHz_counter(counter):
    """
    Convert the tib 10MHz counter to uint32

    The counter is stored using 3 uint8 values forming a 24-bit unsigned integer
    """
    return counter[0] + (counter[1] << 8) + (counter[2] << 16)
