"""Conversions between ground frame and earth location

This will be included in ctapipe 0.19, remove when upgrading.
"""
from astropy.coordinates import AltAz, ITRS, CartesianRepresentation
from ctapipe.coordinates import GroundFrame

def _altaz_to_earthlocation(altaz):
    local_itrs = altaz.transform_to(ITRS(location=altaz.location))
    itrs = ITRS(local_itrs.cartesian + altaz.location.get_itrs().cartesian)
    return itrs.earth_location


def _earthlocation_to_altaz(location, reference_location):
    # See
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html#altaz-calculations-for-earth-based-objects
    # for why this is necessary and we cannot just do
    # `get_itrs().transform_to(AltAz())`
    itrs_cart = location.get_itrs().cartesian
    itrs_ref_cart = reference_location.get_itrs().cartesian
    local_itrs = ITRS(itrs_cart - itrs_ref_cart, location=reference_location)
    return local_itrs.transform_to(AltAz(location=reference_location))

def ground_frame_to_earth_location(ground_frame, reference_location):
    # in astropy, x points north, y points east, so we need a minus for y.
    cart = CartesianRepresentation(ground_frame.x, -ground_frame.y, ground_frame.z)
    altaz = AltAz(cart, location=reference_location)
    return _altaz_to_earthlocation(altaz)

def ground_frame_from_earth_location(location, reference_location):
    altaz = _earthlocation_to_altaz(location, reference_location)
    x, y, z = altaz.cartesian.xyz
    # in astropy, x points north, y points east, so we need a minus for y.
    return GroundFrame(x=x, y=-y, z=z)
