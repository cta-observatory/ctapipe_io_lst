import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation

N_GAINS = 2
N_MODULES = 265
N_PIXELS_MODULE = 7
N_PIXELS = N_MODULES * N_PIXELS_MODULE
N_CAPACITORS_CHANNEL = 1024
# 4 drs4 channels are cascaded for each pixel
N_CAPACITORS_PIXEL = 4 * N_CAPACITORS_CHANNEL
N_SAMPLES = 40
HIGH_GAIN = 0
LOW_GAIN = 1
CLOCK_FREQUENCY_KHZ = 133e3

# we have 8 channels per module, but only 7 are used.
N_CHANNELS_MODULE = 8

# First capacitor order according Dragon v5 board data format
CHANNEL_ORDER_HIGH_GAIN = [0, 0, 1, 1, 2, 2, 3]
CHANNEL_ORDER_LOW_GAIN = [4, 4, 5, 5, 6, 6, 7]

PIXEL_INDEX = np.arange(N_PIXELS)

#: location of lst-1 as `~astropy.coordinates.EarthLocation`
#: Taken from Abelardo's Coordinates of LST-1 & MAGIC presentation
#: https://redmine.cta-observatory.org/attachments/65827
LST1_LOCATION = EarthLocation(
    lon=-17.89149701 * u.deg,
    lat=28.76152611 * u.deg,
    # height of central pin + distance from pin to elevation axis
    height=2184 * u.m + 15.883 * u.m
)

#: Area averaged position of LST-1, MAGIC-1 and MAGIC-2 (using 23**2 and 17**2 m2)
REFERENCE_LOCATION = EarthLocation(
    lon=-17.890879 * u.deg,
    lat=28.761579 * u.deg,
    height=2199 * u.m,  # MC obs-level
)

LST_LOCATIONS = {
    1: LST1_LOCATION,
}
