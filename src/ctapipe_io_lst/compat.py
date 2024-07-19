from ctapipe import __version__ as ctapipe_version

CTAPIPE_VERSION = tuple(int(v) for v in ctapipe_version.split(".")[:3])
CTAPIPE_GE_0_20 = CTAPIPE_VERSION >= (0, 20)
CTAPIPE_GE_0_21 = CTAPIPE_VERSION >= (0, 21)
