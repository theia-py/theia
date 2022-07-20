from .imager import SBMap, plot_circle
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np 
from skimage.measure import block_reduce

class NII_Halpha():
    """
    Container for line ratios built from SB maps. 
    """
    def __init__(self,fof_group,line_1='nii',line_2='ha'):
        """
        Use SBMap to load two SB maps from disk and construct their line ratio. 

        Parameters
        ----------
        fof_group: int
            group catalog ID number of the halo to load 
        line_1: str
            line that will be the numerator of the line ratio. Options include 'nii' and 'ha'.
            (And will one day have oiii too).
        line_2: str
            line that will be the denominator of the line ratio. Options include 'nii' and 'ha'.
        """
        self.fof_group = fof_group
        self.line1 = line_1
        self.line2 = line_2 
        self.nii = SBMap(fof_group,line_1)
        self.halpha= SBMap(fof_group,line_2)
        self.line_ratio_intrinsic = {'face':np.log10(self.nii.map_face/self.halpha.map_face),
                                    'edge':np.log10(self.nii.map_edge/self.halpha.map_edge)}

    def plot_intrinsic_line_ratio(self,plot_re_multiples=[5,10],plot_rvir=True,vmin=-2.0,vmax=1.0,cmap='magma'):
        fig, ax = plt.subplots(1,2,figsize=(21.5,10))
        ax[0].set_aspect(1)

        box_half = self.halpha.boxwidth / 2.0
        im0 = ax[0].imshow(self.line_ratio_intrinsic['edge'],origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=[-box_half,box_half,-box_half,box_half])
        im1 = ax[1].imshow(self.line_ratio_intrinsic['face'],origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=[-box_half,box_half,-box_half,box_half])

        ax[1].set_aspect(1)
        ax1_divider = make_axes_locatable(ax[1])
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cb1 = fig.colorbar(im1, cax=cax1)
        cb1.set_label(r"log([NII] / H$\alpha$)",fontsize=22)
        plt.subplots_adjust(wspace=0.0)
        ax[1].set_yticks([])
        #ax[1].set_xticks([-200,-100,0,100,200])
        ax[1].set_yticklabels([])
        cax1.tick_params(labelsize=20)
        ax[0].tick_params(labelsize=20)
        ax[1].tick_params(labelsize=20)

        ax[0].set_ylabel('size [kpc]',fontsize=22)
        ax[1].set_xlabel('size [kpc]',fontsize=22)
        ax[0].set_xlabel('size [kpc]',fontsize=22)
        

        plot_circle(r=self.halpha.hmr,ax=ax[0],color='w')
        plot_circle(r=self.halpha.hmr,ax=ax[1],color='w')
        if plot_re_multiples is not None:
            for i in plot_re_multiples:
                plot_circle(r=i*self.halpha.hmr,ax=ax[0],color='w')
                plot_circle(r=i*self.halpha.hmr,ax=ax[1],color='w')
        if plot_rvir:
            plot_circle(r=self.halpha.rvir,ax=ax[0],color='k')
            plot_circle(r=self.halpha.rvir,ax=ax[1],color='k')

        for i in ax:
            i.set_xlim(-box_half,box_half)
            i.set_ylim(-box_half,box_half)
        return fig, ax


    def setup_halpha_imager(self,**kwargs):
        self.halpha.setup_instrument(**kwargs)
    def setup_nii_imager(self,**kwargs):
        self.nii.setup_instrument(**kwargs)
    def simulate_halpha(self,**kwargs):
        self.halpha.simulate_on_detector(**kwargs)
    def simulate_nii(self,**kwargs):
        self.nii.simulate_on_detector(**kwargs)
    def visualize_on_detector(self,
                            log=True,
                            plot_re_multiples=None,
                            plot_rvir=False,
                            cmap='magma',
                            vmin=None,
                            vmax=None,
                            binning_factor=None,
                            fill='white',
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
        if binning_factor is not None:
            Ha_map_edge = block_reduce(self.halpha.map_edge_observed, block_size=(binning_factor, binning_factor), func=np.mean)
            Ha_map_face = block_reduce(self.halpha.map_face_observed, block_size=(binning_factor, binning_factor), func=np.mean)
            NII_map_edge = block_reduce(self.nii.map_edge_observed, block_size=(binning_factor, binning_factor), func=np.mean)
            NII_map_face = block_reduce(self.nii.map_face_observed, block_size=(binning_factor, binning_factor), func=np.mean)
        
            map_edge = NII_map_edge / Ha_map_edge
            map_face = NII_map_face / Ha_map_face
        else:
            map_edge = self.nii.map_edge_observed / self.halpha.map_edge_observed
            map_face = self.nii.map_face_observed / self.halpha.map_face_observed
        if log:
            map_edge = np.log10(map_edge)
            map_face = np.log10(map_face)
            if fill == 'black':
                map_edge[np.isnan(map_edge)] = vmin-10
                map_face[np.isnan(map_face)] = vmin-10

        
        
        if vmin is None:
            vmin = np.mean(map_edge) - np.std(map_edge)
        if vmax is None:
            vmax = np.mean(map_edge) + np.std(map_edge)
        bb_context = plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'})
        wb_context = plt.rc_context({'axes.edgecolor':'black', 'xtick.color':'black', 'ytick.color':'black', 'figure.facecolor':'white'})
        if context=='black':
            cont = bb_context 
        else:
            cont = wb_context
        with cont:
            fig, ax = plt.subplots(1,2,figsize=(21.5,10))
            ax[0].set_aspect(1)
        
            im0 = ax[0].imshow(map_edge,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=[-self.nii.box_half,self.nii.box_half,-self.nii.box_half,self.nii.box_half])
            im1 = ax[1].imshow(map_face,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=[-self.nii.box_half,self.nii.box_half,-self.nii.box_half,self.nii.box_half])

            ax[1].set_aspect(1)
            ax1_divider = make_axes_locatable(ax[1])
            cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            cb1 = fig.colorbar(im1, cax=cax1)
            if context =='black':
                if log:
                    cb1.set_label(r"log ([NII] / H$\alpha$)",fontsize=22,color='white')
                else:
                    cb1.set_label(r"[NII] / H$\alpha$",fontsize=22,color='white')
            else:
                if log:
                    cb1.set_label(r"log ([NII] / H$\alpha$)",fontsize=22,color='black')
                else:
                    cb1.set_label(r"[NII] / H$\alpha$",fontsize=22,color='black')
            plt.subplots_adjust(wspace=0.0)
            ax[1].set_yticks([])
            #ax[1].set_xticks([-200,-100,0,100,200])
            ax[1].set_yticklabels([])
            cax1.tick_params(labelsize=20)
            ax[0].tick_params(labelsize=20)
            ax[1].tick_params(labelsize=20)

            ax[0].set_ylabel('size [pixels]',fontsize=22)
            ax[1].set_xlabel('size [pixels]',fontsize=22)
            ax[0].set_xlabel('size [pixels]',fontsize=22)
            

            plot_circle(r=self.nii.hmr_in_pix,ax=ax[0],color='w')
            plot_circle(r=self.nii.hmr_in_pix,ax=ax[1],color='w')
            if plot_re_multiples is not None:
                for i in plot_re_multiples:
                    plot_circle(r=i*self.nii.hmr_in_pix,ax=ax[0],color='w')
                    plot_circle(r=i*self.nii.hmr_in_pix,ax=ax[1],color='w')
            if plot_rvir:
                plot_circle(r=self.nii.rvir_in_pix,ax=ax[0],color='k')
                plot_circle(r=self.nii.rvir_in_pix,ax=ax[1],color='k')

            for i in ax:
                i.set_xlim(-self.nii.box_half,self.nii.box_half)
                i.set_ylim(-self.nii.box_half,self.nii.box_half)
                if context == 'black':
                    i.xaxis.label.set_color('white')
                    i.yaxis.label.set_color('white')

            return fig, ax