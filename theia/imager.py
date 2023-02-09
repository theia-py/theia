
from copy import deepcopy

# Third-party
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy import constants


from astropy import units as u
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import FancyBboxPatch
from astropy.convolution import convolve_fft
import matplotlib
from .mags_to_counts import mags_to_counts
from skimage.measure import block_reduce
matplotlib.rcParams['axes.linewidth'] = 1.5 #set the value globally
from reproject import reproject_adaptive
from astropy.wcs import WCS

import copy 
import asdf 
from .sky import calculate_sky_counts
from .utils import gaussian_psf, plot_circle, get_angular_extent, check_units
from .TNG50.tng_utils import load_TNG_galaxy

import jax
import jax.numpy as jnp
from jax import  jit 
import jax.scipy as jsp

class SBMap():
    """
    Container for SB maps, to perform observing simulations.
    """
    def __init__(self):
        """
        Initialize Object.

        Parameters
        ----------
        fof_group: int
            ID of the FOF group in TNG50 to load. 
        line: str
            emission line to load. options: ha, nii
        """
        pass 

    def load_TNG50(self,fof_group,line,fof_path,cat_path,orientation='face'):
        self.fof_group = fof_group 
        map_edge, map_face = load_TNG_galaxy(fof_group,line,fof_path=fof_path)

        with asdf.open(cat_path) as af:
            self.gal_props = af.tree[fof_group]
            print(self.gal_props)
            self.rvir = check_units(self.gal_props['rvir'],'kpc')
            self.boxwidth = 2.0*self.rvir 
            self.hmr = check_units(self.gal_props['stellar_hmr'],'kpc')
        if orientation =='face':
            self.map = map_edge 
        elif orientation == 'edge':
            self.map = map_edge

    def convert_energy_to_photon(self,image,wavelength_emit):
        wavelength_emit = check_units(wavelength_emit,'angstrom')
        im = image * u.erg/u.s/u.cm**2/u.arcsec**2 
        photon_energy = constants.h * constants.c / (wavelength_emit)

        im /= photon_energy.to(u.erg)
        im = (im*u.photon).to(u.photon/u.s/u.cm**2/u.arcsec**2) 
        return im.value
        

    def add_images(self,image,box_length_kpc,rvir=None,stellar_hmr=None):
        self.rvir = rvir 
        self.hmr = stellar_hmr 
        self.map = image
        self.boxwidth=check_units(box_length_kpc,'kpc')

    def plot_map(self,scale='log',plot_re_multiples=None,plot_rvir=False,vmin=5e-14,vmax=5e-6):
        """
        Plot the intrinsic maps from TNG50. 

        Parameters
        ----------
        plot_re_multiples: list, default:None
            list containing multiples of the stellar half mass radius to plot as circles over the map. e.g., [5,10].
        plot_rvir: bool, default: False
            whether to plot a circle at the virial radius of the system. 
        vmin: float, default=5e-14
            vmin for image scaling. If None, will be mu(image) - 2*sigma(image)
        vmax: float, default=5e-6
            vmax for image scaling. If None, will be mu(image) + 2*sigma(image)
        
        Returns
        -------
        fig, ax: `matplotlib.figure`, `matplotlib.axes`
            the figure and axes objects for further manipulation.
        """
        fig, ax = plt.subplots(figsize=(15,15))
        ax.set_aspect(1)
        if vmin is None:
            vmin = np.mean(self.map_edge) - 2*np.std(self.map_edge)
        if vmax is None:
            vmax = np.mean(self.map_edge) + 2*np.std(self.map_edge)
        box_half = self.boxwidth / 2.0
        box_half = box_half.value
        if scale=='log':
            im0 = ax.imshow(self.map,origin='lower',cmap='gray_r',norm=LogNorm(vmin=vmin,vmax=vmax),extent=[-box_half,box_half,-box_half,box_half])
        elif scale=='linear':
            im0 = ax.imshow(self.map,origin='lower',cmap='gray_r',vmin=vmin,vmax=vmax,extent=[-box_half,box_half,-box_half,box_half])

        ax1_divider = make_axes_locatable(ax)
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cb1 = fig.colorbar(im0, cax=cax1)
        cb1.set_label(r"Surface Brightness [photon s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]",fontsize=22)
        plt.subplots_adjust(wspace=0.0)
        ax.set_yticks([])
        ax.set_yticklabels([])
        cax1.tick_params(labelsize=20)
        ax.tick_params(labelsize=20)
        ax.set_ylabel('size [kpc]',fontsize=22)
        ax.set_xlabel('size [kpc]',fontsize=22)
        
        if plot_re_multiples is not None:
            plot_circle(r=self.hmr,ax=ax,color='w')
            for i in plot_re_multiples:
                plot_circle(r=i*self.hmr,ax=ax,color='w')
        if plot_rvir:
            plot_circle(r=self.rvir,ax=ax,color='k')


        ax.set_xlim(-box_half,box_half)
        ax.set_ylim(-box_half,box_half)
        return fig, ax
    
    def on_sky(self,distance,plot_re_multiples=None,plot_rvir=False,vmin=5e-14,vmax=5e-6,context='white'):
        """
        Plot the intrinsic maps from TNG50 placed at a certain distance, now in angular on-sky units (arcmin).

        Parameters
        ----------
        distance: float
            distance to place galaxy at. Should be in Mpc. 
        plot_re_multiples: list, default:None
            list containing multiples of the stellar half mass radius to plot as circles over the map. e.g., [5,10].
        plot_rvir: bool, default: False
            whether to plot a circle at the virial radius of the system. 
        vmin: float, default=5e-14
            vmin for image scaling. If None, will be mu(image) - 2*sigma(image)
        vmax: float, default=5e-6
            vmax for image scaling. If None, will be mu(image) + 2*sigma(image)
        context: str, default: 'white'
            plotting context; 'white' for standard, 'black' for black background with white ticks and labels.
        Returns
        -------
        fig, ax: `matplotlib.figure`, `matplotlib.axes`
            the figure and axes objects for further manipulation.
        """
        distance = check_units(distance,'Mpc')
        if vmin is None:
            vmin = np.mean(self.map) - np.std(self.map)
        if vmax is None:
            vmax = np.mean(self.map_) + np.std(self.map)
        boxwidth = get_angular_extent(self.boxwidth,distance).value
            
        
        box_half = boxwidth / 2.0
        
        bb_context = plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'})
        wb_context = plt.rc_context({'axes.edgecolor':'black', 'xtick.color':'black', 'ytick.color':'black', 'figure.facecolor':'white'})
        if context=='black':
            cont = bb_context 
        else:
            cont = wb_context
        with cont:
            fig, ax = plt.subplots(figsize=(15,15))
            ax.set_aspect(1)
            
            
            im0 = ax.imshow(self.map,origin='lower',cmap='gray_r',norm=LogNorm(vmin=vmin,vmax=vmax),extent=[-box_half,box_half,-box_half,box_half])


            ax1_divider = make_axes_locatable(ax)
            cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            cb1 = fig.colorbar(im0, cax=cax1)
            if context=='black':
                cb1.set_label(r"Surface Brightness [photon s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]",fontsize=22,color='white')
            else:
                cb1.set_label(r"Surface Brightness [photon s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]",fontsize=22,color='black')
            plt.subplots_adjust(wspace=0.0)

            cax1.tick_params(labelsize=20)
            ax.tick_params(labelsize=20)

            ax.set_ylabel('size [arcmin]',fontsize=22)
            ax.set_xlabel('size [arcmin]',fontsize=22)
                        
            if plot_re_multiples is not None:
                hmr_in_pix = get_angular_extent(self.hmr*2,distance).value /2.0
                rvir_in_pix = get_angular_extent(self.rvir*2,distance).value/2.0 
                plot_circle(r=hmr_in_pix,ax=ax,color='w')
                for i in plot_re_multiples:
                    plot_circle(r=i*hmr_in_pix,ax=ax,color='w')
            if plot_rvir:
                plot_circle(r=rvir_in_pix,ax=ax,color='k')
            ax.set_xlim(-box_half,box_half)
            ax.set_ylim(-box_half,box_half)
            if context == 'black':
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')

            return fig, ax
    def setup_instrument(self,
                        optical_diameter,
                        pixel_scale,
                        source_velocity,
                        emission_wl,
                        read_noise,
                        efficiency,
                        dark_current,
                        filter_bandpass=8*u.angstrom,
                        **kwargs):
        """
        Setup an instrument by providing all of the relevant details about it.

        Parameters
        ----------
        optical_diameter: int or `~astropy.units.Quantity`
            diameter of the optical system being used, in meters.
        pixel_scale: float
            pixel scale in arcsec/pixel (e.g., 2.1).
        source_velocity: float
            velocity of the source (in km/s). Our filters will go from 0 to 4500 km/s, but the code
            will not stop you from entering any particular value. This parameter sets the lambda_eff
            of the bandpass, which controls the sky level when using UVES spectrum. 
        emission_wl: float or `~astropy.units.Quantity`
            intrinsic emission wavelength of the line in the map you are using in Angstrom. E.g., for H-alpha this
            would be 65652.3. This plus the source velocity sets the ultimate bandpass location.
        read_noise: float
            read noise of the detector in e-. 
        efficiency: float
            quantum efficiency of the detector *at the wavelengths of interest*. Between 0 and 1. 
        dark_current: float
            dark current in e- per second for the operating cooled detector. 
        filter_bandpass: float or `~astropy.units.Quantity`, default: 8*u.angstrom
            filter bandpass as an astropy length quantity (e.g., u.nm or u.angstrom, assumed angstrom)
        
        """
        self.diameter = check_units(optical_diameter,'m')
        self.pixel_scale= pixel_scale
        self.area = np.pi*(self.diameter/2.0)**2
        self.read_noise = read_noise
        self.efficiency = efficiency
        self.dark_current = dark_current
        self.filter_bandpass = check_units(filter_bandpass,'angstrom')
        emission_wl = check_units(emission_wl,'angstrom')
        source_velocity = check_units(source_velocity,'km/s')
        self.lam_eff = self.lam_eff_from_v(emission_wl,source_velocity)
        self.storage = {
                        'pixscale':f'{pixel_scale} "/pix',
                        'area': f'{self.area:.2f}',
                        'rdnoise': f'{read_noise} e-',
                        'thruput': f'{efficiency}',
                        'darkcurr': f'{dark_current} e-/s',
                        'bandpass': f'{filter_bandpass.to(u.nm)}',
                        'srcvel': f'{source_velocity}',
                        'srclam': f'{emission_wl}',
                        'lameff': f'{self.lam_eff:.2f}',
        }


    def lam_eff_from_v(self,wavelength,velocity): 
        """
        Determines the center of the bandpass for a source of emission wavelength at some velocity.

        Parameters
        ----------
        wavelength: float
            wavelength of emission, in angstrom
        velocity: float
            velocity of source, in km/s. 
        
        Returns
        ------
        lam_eff: `astropy.Quantity`
            center of the bandpass. 
        """
        velocity = velocity
        lam_eff = wavelength * (1+(velocity/constants.c))
        return lam_eff
    
    def convert_maps_to_counts(self,sb_map,exptime):
        """
        Convert a map in ph/s/cm2/arcsec2 into photons, by multipying by exptime and telescope properties.

        Parameters
        ----------
        sb_map: array_like
            the map array in ph/s/cm2/arcsec2
        exptime: float
            exposure time in seconds. 
        
        Returns
        -------
        sb_map: array_like
            the same map, but now in units of photon (over some exptime) arriving at the top of the telescope.
        """
        sb_map = sb_map * u.photon / u.s / u.cm**2 / u.arcsec**2 
        sb_map = sb_map * (exptime*u.s)
        sb_map = sb_map * self.area.to(u.cm**2)
        sb_map = sb_map * (self.pixel_scale*u.arcsec)**2
        return sb_map 
    
    def apply_seeing(self,image,seeing_fwhm,boundary='center'):
        """
        Apply astronomical seeing (PSF) to an object (as a gaussion). 

        Parameters
        ---------
        image: array_like
            image to apply seeing to.
        seeing_fwhm: float
            the astronomical seeing FHWM, in arcseconds.
        boundary: str, default: 'center'
            deals with the fft convolution in the case that your image is not odd. See `convolve_fft`. 
        """
        self.storage['seeing'] = f'{seeing_fwhm} arcsec'
        psf = gaussian_psf(seeing_fwhm,self.pixel_scale)
        image = convolve_fft(image ,
                            psf,
                            boundary=boundary,
                            normalize_kernel=True)
        return image

    def reproject_on_detector(self,
                    distance,
                    exptime,
                    verbose=False,
                    ):
        """
        Simulate the image that would be detected by the telescope. 

        Parameters
        ----------
        distance: float or `~astropy.units.Quantity`
            distance to place the simulated galaxy. In Mpc.
        exptime: float or `~astropy.units.Quantity`
            exposure time (for a single science exposure). In seconds.
        n_exposures: int
            number of science exposures to be combined in a final stack.
        combine: float = 'mean'
            how to combine the n exposures. 'mean' or 'median' supported.
        n_proc: int, default:4
            multi-threaded RNG parameter for number of processors to use.
        seed: int, default: None
            seed for RNG repeatability if desired.
        use_sky_spectrum: bool, default=True
            if True, integrate a 1-bandpass chunk of the UVES sky spectrum at lam_eff to estimate
            the sky counts.
        sky_magnitude: float, default: None
            If use_sky_spectrum is False, supply a sky magnitude (e.g., 21.8) to use to estimate counts.
        verbose: bool, default: False
            simulation can take a while as it involves reprojection and a lot of RNG. Print steps along the way.
        """
        exptime = check_units(exptime,'s')
        self.exptime = exptime
        distance = check_units(distance,'Mpc')
        self.distance=distance
        self.storage['exptime'] = f'{exptime}'
        self.storage['D'] = f'{distance}'
        
        
        pixel_scale = self.pixel_scale
        boxwidth = get_angular_extent(self.boxwidth,distance).to(u.arcsec).value
        n_pix_needed = int(boxwidth / self.pixel_scale) 
        detector_dims = (n_pix_needed,n_pix_needed)

        current_pixel_scale = (boxwidth*u.arcsec) / (self.map.shape[0]*u.pixel)
        desired_pixel_scale = pixel_scale*u.arcsec / u.pixel 
        self.boxwidth_in_pix = ((boxwidth*u.arcsec) / desired_pixel_scale).value
        
        
        self.box_half = self.boxwidth_in_pix / 2.0
        input_wcs = WCS(naxis=2)
        input_wcs.wcs.crpix = 0,0
        input_wcs.wcs.cdelt = -current_pixel_scale.value, current_pixel_scale.value

        output_wcs = WCS(naxis=2)
        output_wcs.wcs.crpix = 0,0
        output_wcs.wcs.cdelt = -desired_pixel_scale.value, desired_pixel_scale.value
        
        #Modify maps into counts
        map_use= self.convert_maps_to_counts(self.map,exptime)*self.efficiency
        
        if verbose:
            print('Reprojecting onto DSLM pixel scale.')

        self.reprojected_map = reproject_adaptive((map_use,input_wcs),output_wcs,shape_out=detector_dims,conserve_flux=True,boundary_mode='nearest')
        
        
        
    def simulate_images(self,
                        n_exposures:int,
                        radii:bool=True,
                        verbose:bool=True,
                        use_sky_spectrum=True,
                        sky_magnitude = None,
                        sky_counts = None,
                        seeing_fwhm=None,
                        seed=None):
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key =jax.random.PRNGKey(0)
        pixel_scale = self.pixel_scale 
        if radii:
            hmr_in_arcsec = get_angular_extent(self.hmr*2.0,self.distance).to(u.arcsec).value/2.0
            self.hmr_in_pix = hmr_in_arcsec / pixel_scale 
            rvir_in_arcsec = get_angular_extent(self.rvir*2.0,self.distance).to(u.arcsec).value/2.0
            self.rvir_in_pix = rvir_in_arcsec / pixel_scale     
        
        if verbose:
            print('Sampling noise.')
        if use_sky_spectrum:
            sky_counts = calculate_sky_counts(self.lam_eff,
                                            self.filter_bandpass,
                                            self.exptime*u.s,
                                            pixel_scale=self.pixel_scale*u.arcsec,
                                            effective_area=self.area
                                            )
        else:
            if sky_magnitude is not None:
                sky_counts = mags_to_counts(sky_magnitude,
                                            self.exptime,
                                            self.pixel_scale*(u.arcsec/u.pixel),
                                            self.area,
                                            self.efficiency,
                                            self.filter_bandpass,
                                            self.lam_eff)
            elif sky_counts is not None:
                sky_counts = sky_counts
            else: 
                raise AssertionError('Either sky counts or sky mag must be provided if use spectrum is false.')
        map_use = self.reprojected_map
        map_out = jnp.zeros(map_use.shape)
        dark_current_total = self.dark_current*self.exptime.value
        
        for i in tqdm(range(int(n_exposures/10))):
            key, subkey = jax.random.split(key)
            key, subkey2 = jax.random.split(key)
            rdnoise_map = jax.random.normal(key=subkey,shape=(map_out.shape[0],map_out.shape[1],10))*self.read_noise
            map_counts = map_use + sky_counts + dark_current_total
            map_counts = jnp.repeat(map_counts[:, :, np.newaxis], 10, axis=2)
            map_observed = jax.random.poisson(key=subkey2,lam=map_counts) - sky_counts - dark_current_total

            map_out = map_out + jnp.sum(map_observed + rdnoise_map, axis=-1)
        if verbose:
            print('Combining individual exposures.')
        self.map_observed = map_out / n_exposures 

        if seeing_fwhm is not None:
            self.map_observed = self.apply_seeing(self.map_observed,seeing_fwhm)

    
    def visualize_on_detector(self,
                            plot_re_multiples=None,
                            plot_rvir=False,
                            vmin=None,
                            vmax=None,
                            binning_factor=None,
                            overlay_properties=True,
                            context='white'):
        """
        Plot the observed maps from TNG50 placed at a certain distance, now in detector pixels with all noise added.

        Parameters
        ---------- 
        plot_re_multiples: list, default:None
            list containing multiples of the stellar half mass radius to plot as circles over the map. e.g., [5,10].
        plot_rvir: bool, default: False
            whether to plot a circle at the virial radius of the system. 
        vmin: float, default=5e-14
            vmin for image scaling. If None, will be mu(image) - 2*sigma(image)
        vmax: float, default=5e-6
            vmax for image scaling. If None, will be mu(image) + 2*sigma(image)
        binning_factor: int, default: None
            whether to bin up pixels and take their mean before display.
        context: str, default: 'white'
            plotting context; 'white' for standard, 'black' for black background with white ticks and labels.
        Returns
        -------
        fig, ax: `matplotlib.figure`, `matplotlib.axes`
            the figure and axes objects for further manipulation.
        """
        map_plot = copy.deepcopy(self.map_observed)
        
        if binning_factor is not None:
            map_plot = block_reduce(map_plot, block_size=(binning_factor, binning_factor), func=np.mean)
        
        if vmin is None:
            vmin = np.mean(map_plot) - np.std(map_plot)
        if vmax is None:
            vmax = np.mean(map_plot) + np.std(map_plot)
        bb_context = plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'})
        wb_context = plt.rc_context({'axes.edgecolor':'black', 'xtick.color':'black', 'ytick.color':'black', 'figure.facecolor':'white'})
        if context=='black':
            cont = bb_context 
        else:
            cont = wb_context
        with cont:
            fig, ax = plt.subplots(figsize=(15,15))
            ax.set_aspect(1)
            minx = -self.box_half
            maxx = self.box_half
            miny = -self.box_half 
            maxy = self.box_half

            im0 = ax.imshow(map_plot,origin='lower',cmap='gray_r',vmin=vmin,vmax=vmax,extent=[minx,maxx,miny,maxy])
            ax1_divider = make_axes_locatable(ax)
            cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            cb1 = fig.colorbar(im0, cax=cax1)
            if context =='black':
                cb1.set_label(r"Counts in exptime",fontsize=22,color='white')
            else:
                cb1.set_label(r"Counts in exptime",fontsize=22,color='black')

            cax1.tick_params(labelsize=20)
            ax.tick_params(labelsize=20)

            ax.set_ylabel('size [pixels]',fontsize=22)
            ax.set_xlabel('size [pixels]',fontsize=22)
            

            
            if plot_re_multiples is not None:
                plot_circle(r=self.hmr_in_pix,ax=ax,color='w')
                for i in plot_re_multiples:
                    plot_circle(r=i*self.hmr_in_pix,ax=ax,color='w')
            if plot_rvir:
                plot_circle(r=self.rvir_in_pix,ax=ax,color='k')


            ax.set_xlim(-self.box_half,self.box_half)
            ax.set_ylim(-self.box_half,self.box_half)

            if context == 'black':
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')

            if overlay_properties:
                if hasattr(self,'gal_props'):
                    rect = FancyBboxPatch((0.65, 0.025), 0.33,0.18, boxstyle="round,pad=0.01",linewidth=1, edgecolor='k', facecolor='white',transform=ax.transAxes,zorder=10,alpha=0.85)
                    ax.add_patch(rect)
                    ax.text(0.95,0.15,rf"log $M_*$: {self.gal_props['subhalo_stellar_mass']:.2f}",color='k',fontsize=20,transform=ax.transAxes,ha='right',zorder=11)
                    ax.text(0.95,0.05,rf"$R_{{vir}}$: {self.gal_props['rvir']:.2f}",color='k',fontsize=20,transform=ax.transAxes,ha='right',zorder=11)

            return fig, ax
    def save_fits(self,fname,binning_factor=None):
        '''
        Save a fits file of the current "observed" map. 
        The Header will contain information about the simulated image.
        The Primary HDU is empty, and extensions 1 and 2 contain the face on and edge on maps.

        Parameters
        ----------
        fname: str
            filepath to save the fits file.
        '''
        map_edge = copy.deepcopy(self.map_edge_observed)
        map_face = copy.deepcopy(self.map_face_observed)
        if binning_factor is not None:
            map_edge = block_reduce(map_edge, block_size=(binning_factor, binning_factor), func=np.mean)
            map_face = block_reduce(map_face, block_size=(binning_factor, binning_factor), func=np.mean)

        hdr = fits.Header()
        for i in self.storage.keys():
            hdr[i] = self.storage[i]
        hdr['EXT1'] = 'MAP FACE'
        hdr['EXT2'] = 'MAP EDGE'
        for i in self.storage.keys():
            hdr[i] = self.storage[i]
        hduPrimary = fits.PrimaryHDU(header=hdr)
        hdu1 = fits.ImageHDU(map_face)
        hdu2 = fits.ImageHDU(map_edge)
        hdu_out = fits.HDUList([hduPrimary,hdu1,hdu2])
        hdu_out.writeto(fname,overwrite=True)




        


