from ctapipe import __version__ as ctapipe_version

CTAPIPE_VERSION = tuple(int(v) for v in ctapipe_version.split(".")[:3])
CTAPIPE_GE_0_27 = CTAPIPE_VERSION >= (0, 27)
