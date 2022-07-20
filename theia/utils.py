import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, Moffat2DKernel

def gaussian_psf(fwhm, pixel_scale=0.2, shape=41, mode='center', factor=10):
    """
    Gaussian point-spread function.
    Parameters
    ----------
    fwhm : float or `~astropy.units.Quantity`
        Full width at half maximum of the psf. If a float is given, the units
        will be assumed to be `~astropy.units.arcsec`. The units can be angular 
        or in pixels.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the psf image. If a float is given, the units are 
        assumed to be `~astropy.units.arcsec` per `~astropy.units.pixel` 
        (why would you want anything different?).
    shape : int or list-like, optional
        Shape of the psf image. Must be odd. If an int is given, the x and y 
        dimensions will be set to this value: (shape, shape).
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by linearly interpolating
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin. Very slow.
    factor : number, optional
        Factor of oversampling. Default factor = 10. If the factor
        is too large, evaluation can be very slow.
    Returns
    -------
    psf : `~numpy.ndarray`
        The PSF image normalized such that its sum is equal to one. 
    """
    pixel_scale = pixel_scale*(u.arcsec/u.pixel)
    width = (fwhm*u.arcsec).to('pixel', u.pixel_scale(pixel_scale)).value
    width *= gaussian_fwhm_to_sigma
    x_size, y_size = shape, shape
    model = Gaussian2DKernel(
        x_stddev=width,
        y_stddev=width,
        x_size=x_size,
        y_size=y_size,
        mode=mode,
        factor=factor
    )
    model.normalize()
    psf = model.array
    return psf


def plot_circle(r,ax,color,**kwargs):
    """
    Feeding in an axis, plot a circle in the extent-units of that axis (namely, an image). 

    Parameters
    ----------
    r: float
        radius of the circle to plot.
    ax: `matplotlib.axes`
        axis on which to plot the circle
    color: str
        color to make the cirle
    **kwargs
        any kwargs recognized by `ax.plot()`.
    """
    arr = np.linspace(0,2*np.pi,100)
    x = r*np.sin(arr)
    y = r*np.cos(arr)
    ax.plot(x,y,ls='-',color=color,lw=3)

def get_angular_extent(boxwidth,distance):
    """
    Given a linear extent on the sky in physical units (e.g., kpc), determine the 
    angular extent (in arcmin).

    Parameters
    ----------
    boxwidth: float
        full width of a physical box (or line). Should be in kpc.
    distance: float
        distance to place the box/line. Should be in Mpc.
    
    Returns
    angular_size: `astropy.Quantity`
        angle on sky corresponding to the full length, in arcminutes.
    """
    boxhalf = boxwidth/2.0
    distance = distance*u.Mpc
    boxhalf = boxhalf*u.kpc
    angular_size = np.arctan(boxhalf/distance).to(u.arcmin)
    return angular_size*2