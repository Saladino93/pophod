"""
Utility functions.
"""

import numpy as np



def get_nz(redshifts: np.ndarray, weights: np.ndarray = None, nbins: int = 20, density: bool = False, getbins: bool = False):
    """
    Get some dndz by binning some redshifts.
    """
    galaxyzbins = np.linspace(redshifts.min(), redshifts.max(), nbins) if type(nbins) == int else nbins
    #CREATE dn/dz
    histog = np.histogram(redshifts, galaxyzbins, weights = weights, density = density)
    bins, nz = histog[1], histog[0]
    z = (bins[1:]+bins[:-1])/2.
    if getbins:
        return z, nz, bins
    else:
        return z, nz