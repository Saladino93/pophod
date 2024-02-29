"""
Just cut the websky catalog to the desired redshift range and save it to a new file.
"""

import numpy as np
import pathlib
from pophod.utils import websky


zcouples = [(0., 0.7), (0., 0.8), (0., 1), (0.4, 1.2), (0.5, 1.6), (0.5, 1.6)]

direc = '/Volumes/Omar T7 Shield/Websky Sims'
directory = pathlib.Path(direc)
websky_halo_catalog_name = 'halos.pksc'


W = websky.WebSky(directory_path = direc, websky_version = "")
Nmax = None
Nmax = 5e8

for zmin, zmax in zcouples:
    try:
        matrix = np.load(f"matrix_websky_zmin_{zmin}_zmax{zmax}.npy")
    except:
        name = f"../data/matrix_websky_zmin_{zmin}_zmax{zmax}.npy"
        matrix = W.load_halo_catalogue(zmin = zmin, zmax = zmax, Nmax = Nmax)
        np.save(name, matrix)