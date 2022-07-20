import numpy as np 
import matplotlib.pyplot as plt 
import astropy.units as u 
from astropy import constants 
from astropy.io import fits 
from glob import glob 
from scipy.integrate import simpson 


def load_uves(fits_dir: str = 'UVES_sky_all/'):
    filelist = glob(fits_dir+'/*.fits')
    print(filelist)
    wl_list = {}
    speclist = {}
    for i in filelist:
        with fits.open(i) as hdu:
            spec = hdu[0].data * 1e-16 * u.erg/u.s/u.cm**2/u.angstrom/u.arcsec**2
            h = hdu[0].header 
            wl_min = h['CRVAL1']
            wl_len = len(spec)
            wl_delt = h['CDELT1']
            wl_max = wl_min + wl_delt*wl_len 
            wl_arr = np.arange(wl_min,wl_max,wl_delt)*u.angstrom
            key = i.split('_')[-1].split('.')[0]
            wl_list[key] = wl_arr
            speclist[key] = spec 
            

    return wl_list, speclist


def calculate_sky_counts(lam_effective,filter_bandpass,exptime,pixel_scale,effective_area):
    """
    Calculate the sky brightness is AB magnitudes using UVES sky spectrum and 
    filter wavelength and filter bandpass. 
    """
    wl_dict, spec_dict = load_uves()
    if lam_effective < 6605*u.angstrom:
        key = '580U'
    else:
        key = '860L'
    wl = wl_dict[key]
    spec = spec_dict[key]
    lam_left = lam_effective - (0.5*filter_bandpass)
    lam_right = lam_effective + (0.5*filter_bandpass)

    ind, = np.where((wl>=lam_left)&(wl<=lam_right))

    filt_wl = wl[ind]
    filt_spec = spec[ind]

    # Numerically integrate the spectrum over the bandpass 
    # First convert to photon/s/cm2/arcsec2/AA 
    energy_axis = (constants.h*constants.c/filt_wl).to(u.erg)
    spec_photons = filt_spec / energy_axis 

    integrated_flux = simpson(spec_photons.value,filt_wl.value)
    integrated_flux*= u.photon/u.s/u.cm**2/u.arcsec**2 
    pixel_area = (pixel_scale)**2 
    integrated_photons = integrated_flux * exptime * pixel_area * effective_area
    return integrated_photons.value

