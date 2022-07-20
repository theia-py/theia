
import h5py


def load_TNG_galaxy(fof_group,line,fof_path):
        """
        Load a flat sb map from TNG50, both face on and edge on, and return the images. 

        Parameters
        ----------
        fof_group: int
            ID of the FOF group in TNG50 to load. 
        line: str
            emission line (must be downloaded). currently options are 'ha' and 'nii'. 
        fof_path: str
            path to the location where maps are stored. Inside this folder should contain dirs
            of the form fof_XXX/, inside which any number of /LINE_sb/ (e.g., ha_sb or nii_sb) might be.
        
        Returns
        -------
        edge_on, face_on: array_like
            arrays containing the photon SB in ph/s/cm2/arcsec2 for the edge on and face on orientations.
        """
        ims = []
        for i in [f'fof_{fof_group}/{line}_sb/edge-on.hdf5',f'fof_{fof_group}/{line}_sb/face-on.hdf5']:
            with h5py.File(fof_path+i, 'r') as f:
                phot = 10**f['grid'][:]
                ims.append(phot)
        return ims[0][200:600,200:600],ims[1][200:600,200:600]