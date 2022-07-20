
import sys
import abc
import pickle
from copy import deepcopy

# Third-party
import numpy as np
from astropy.io import fits
from astropy import constants
from astropy import units as u
from astropy.convolution import convolve_fft

mAB_0 = 48.6
def fnu_from_AB_mag(mag):
    """
    Convert AB magnitude into flux density fnu in cgs units.
    """
    fnu = 10.**((mag + mAB_0)/(-2.5))
    return fnu*u.erg/u.s/u.Hz/u.cm**2

def mags_to_counts(sb, exptime, pixel_scale,area,efficiency,dlam,lam_eff):
        """
        Convert a constant surface brightness into counts per pixel.
        Parameters
        ----------
        sb : float
            Surface brightness in units of `~astropy.units.mag` per square
            `~astropy.units.arcsec`.
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        exptime : float or `~astropy.units.Quantity`
            Exposure time. If float is given, the units are assumed to
            be `~astropy.units.second`.
        pixel_scale : `~astropy.units.Quantity`
            Pixel scale.
        Returns
        -------
        counts_per_pixel : float
            Counts per pixel associated with the given surface brightness.
        """

        fnu_per_square_arcsec = fnu_from_AB_mag(sb) / u.arcsec**2
        pixel_scale = pixel_scale.to('arcsec / pixel')
        E_lam = (constants.h * constants.c / lam_eff).decompose().to('erg')
        flam_per_square_arcsec = fnu_per_square_arcsec *\
            constants.c.to('angstrom/s') / lam_eff**2
        flam_per_pixel = flam_per_square_arcsec * pixel_scale**2
        photon_flux_per_sq_pixel = (flam_per_pixel * dlam / E_lam).\
            decompose().to('1/(cm2*pix2*s)')
        counts_per_pixel = photon_flux_per_sq_pixel * exptime.to('s')
        counts_per_pixel *= area.to('cm2') * u.pixel**2
        assert counts_per_pixel.unit == u.dimensionless_unscaled
        counts_per_pixel *= efficiency
        counts_per_pixel = counts_per_pixel.value
        return counts_per_pixel