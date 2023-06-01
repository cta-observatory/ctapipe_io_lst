import pytest
import numpy as np
from ctapipe_io_lst.anyarray_dtypes import parse_tib_10MHz_counter

COUNTER_VALUES = np.array([1, 1234, 2**16 - 1, 2**24 - 1], dtype=np.uint32)


@pytest.mark.parametrize("value", COUNTER_VALUES)
def test_parse_tib_10MHz_counter(value):
    counter_24bit = np.frombuffer(value.tobytes()[:3], np.uint8)
    result = parse_tib_10MHz_counter(counter_24bit)
    assert result == value
