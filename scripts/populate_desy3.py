"""
Script specialised to get DES-Y3 like maps based on the Websky halo catalog. 
"""

from pophod import webskygalaxies as wsg
from pophod.utils import websky, utils

import numpy as np
import pathlib

import healpy as hp

import argparse

parser = argparse.ArgumentParser(description="DES Y3 mocks from Websky halo catalog")
parser.add_argument("-m", "--masked", action = 'store_true', help = "Use to mask.")
parser.add_argument("-b", "--boost", action = 'store_true', help = "Use to boost number of objects.")
parser.add_argument("-s", "--seed", type = int, help = "Seed for the random number generator.", default = 0)
parser.add_argument("-n", "--nside", type = int, help = "Nside for the healpix maps.", default = 2048)
parser.add_argument("-model", "--model", type = str, help = "Model to use for the HOD.", default = "zack")
parser.add_argument("-v", "--version", type = str, help = "Out version.", default = "")

args = parser.parse_args()
masked = args.masked
boost = args.boost
seed = args.seed
nside = args.nside
model = args.model
version = args.version

data_directory = pathlib.Path("../data/")
maps_data = data_directory / "maps"
maps_data.mkdir(exist_ok = True)

zmodel = "zack"
smodelspt = "sanchezspt"
smodelplanck = "sanchezplanck"
models = [zmodel, smodelspt, smodelplanck]
assert model in models, f"Model {model} not recognized. Choose among {models}"

#### DES Y3 info

redshift_bins = [0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05]
indices = range(len(redshift_bins)-1)
indices = [1]

def read_z_nz(zbin):
    return np.loadtxt(nzs_dir/nzs_file(zbin), unpack = True)  
Ngals = [2236462, 1599487, 1627408, 2175171, 1583679, 1494243]

nzs_dir = pathlib.Path('../data/DESY3/dndz/')
nzs_file = lambda zbin: f'galaxy_z_nz_{zbin[0]}-{zbin[1]}.txt'

try:
    gmask_ud = np.load(f"../data/DESY3/gmask_ud_{nside}.npy")
except:
    gmask = np.load("/Volumes/Omar T7 Shield/ACTXDES-DATA/gmask.npy")
    gmask_ud = hp.ud_grade(gmask, nside_out = nside)
    np.save(f"../data/DESY3/gmask_ud_{nside}.npy", gmask_ud)

area_gmask = gmask_ud.mean()*4*np.pi
gmask_ud = gmask_ud.astype(int)

##### MODELS #####


if model == zmodel:

    factorC = 1+0.03
    factorS = 1+0.04
    MaglimHodProps = {}
    MaglimHodProps[0] = {'log10Mmin': 11.75, 'log10M1': 13.36, 'alpha': 1.7, 'sigma_logM': 0.28, "fcen": 1.0}
    MaglimHodProps[1] = {'log10Mmin': 11.93, 'log10M1': 13.43, 'alpha': 1.83, 'sigma_logM': 0.26, "fcen": 1.0}
    MaglimHodProps[2] = {'log10Mmin': 11.88, 'log10M1': 12.84, 'alpha': 1.24, 'sigma_logM': 0.21, "fcen": 1.0}
    MaglimHodProps[3] = {'log10Mmin': 11.82, 'log10M1': 13.44, 'alpha': 2.29, 'sigma_logM': 0.31, "fcen": 1.0}
    a_values = [0.41, 0.63, 1.21, 1.19]

    for M in MaglimHodProps:
        MaglimHodProps[M]['kappa'] = 0.
        MaglimHodProps[M]["log10Mmin"] *= factorC
        MaglimHodProps[M]["log10M1"] *= factorS


    #just using Sánchez et al.
    MaglimHodProps[4] = {'log10Mmin': 12.2, 'log10M1': 12.6, 'alpha': 0.57}
    MaglimHodProps[5] = {'log10Mmin': 12.0, 'log10M1': 12.4, 'alpha': 1.06}
    for M in range(4, 6):
        MaglimHodProps[M]['kappa'] = 1.
        MaglimHodProps[M]["fcen"] = 1.
        MaglimHodProps[M]["sigma_logM"] = 0.15

elif model == smodelspt:
    #uses Sánchez et al., this has also a different formula for HOD
    MaglimHodProps = {}
    MaglimHodProps[0] = {'log10Mmin': 12.1, 'log10M1': 13.2, 'alpha': 0.97}
    MaglimHodProps[1] = {'log10Mmin': 12.1, 'log10M1': 13.1, 'alpha': 0.62}
    MaglimHodProps[2] = {'log10Mmin': 12.1, 'log10M1': 12.9, 'alpha': 0.71}
    MaglimHodProps[3] = {'log10Mmin': 12.1, 'log10M1': 12.7, 'alpha': 0.54}
    MaglimHodProps[4] = {'log10Mmin': 12.2, 'log10M1': 12.6, 'alpha': 0.57}
    MaglimHodProps[5] = {'log10Mmin': 12.0, 'log10M1': 12.4, 'alpha': 1.06}
    a_values = np.ones(6)

    for M in MaglimHodProps:
        MaglimHodProps[M]['kappa'] = 1.
        MaglimHodProps[M]["fcen"] = 1.
        MaglimHodProps[M]["sigma_logM"] = 0.15

elif model == smodelplanck:
    #uses Sánchez et al., this has also a different formula for HOD
    MaglimHodProps = {}
    MaglimHodProps[0] = {'log10Mmin': 12.0, 'log10M1': 12.8, 'alpha': 0.68}
    MaglimHodProps[1] = {'log10Mmin': 12.2, 'log10M1': 13.0, 'alpha': 0.61}
    MaglimHodProps[2] = {'log10Mmin': 11.9, 'log10M1': 13.0, 'alpha': 0.84}
    MaglimHodProps[3] = {'log10Mmin': 12.2, 'log10M1': 12.7, 'alpha': 0.73}
    MaglimHodProps[4] = {'log10Mmin': 12.2, 'log10M1': 12.4, 'alpha': 0.86}
    MaglimHodProps[5] = {'log10Mmin': 12.1, 'log10M1': 12.2, 'alpha': 1.91}
    a_values = np.ones(6)

    for M in MaglimHodProps:
        MaglimHodProps[M]['kappa'] = 1.
        MaglimHodProps[M]["fcen"] = 1.
        MaglimHodProps[M]["sigma_logM"] = 0.15



#### Read Websky info
    
zcouples = [(0., 0.7), (0., 0.8), (0., 1), (0.4, 1.2), (0.5, 1.6), (0.5, 1.6)]

direc = '/Volumes/Omar T7 Shield/Websky Sims'
directory = pathlib.Path(direc)

Web = websky.WebSky(directory_path = direc, websky_version = "")

"""
from actxdes.pipeline.data import des_map_extractor as dme
for i in range(6):
    filename = "/Volumes/Omar T7 Shield/ACTXDES-DATA/mag_lim_lens_sample_combined_jointmask_sample_nbins1d_10_weighted2.0sig_pca_maps_107_50_cut.fits"
    catdes = dme.read_catalog(filename)
    zdes = catdes["DNF_ZMC_SOF"]
    zbins = catdes["DNF_ZMEAN_SOF"]
    weight = catdes["weight"]
    np.savetxt(f"../data/DESY3/dndz/zbins_zdes_weight_{i}.txt", np.c_[zbins, zdes, weight])"""

for i in indices:

    cosmology = Web.websky_cosmo.copy()

    zbins, zdes, weight = np.loadtxt(nzs_dir/f"zbins_zdes_weight_{i}.txt").T
    selection = (zbins > redshift_bins[i]) & (zbins < redshift_bins[i+1])
    zdes = zdes[selection]
    weight = weight[selection]
    zin, nzin, bins = utils.get_nz(zdes, weights = weight, nbins = 100, density = False, getbins = True)

    zmin, zmax = zcouples[i]
    name = f"../data/matrix_websky_zmin_{zmin}_zmax{zmax}.npy"
    matrix = np.load(name)
    print(f"Nhalos = {matrix.shape[0]}")

    position, mass, redshift = matrix[:, 0:3], matrix[:, 6], matrix[:, 7]
    W = wsg.WebSkyGalaxyCatalog(seed, cosmology)
    afactor = a_values[i]

    Ngal_des = Ngals[i]
    nbar = Ngal_des/area_gmask
    Nfullsky = nbar*4*np.pi
    factor = Nfullsky/Ngal_des

    factor = factor if boost else 1
    gmask_ud = gmask_ud if masked else None
    mappa, zs = W.from_cat_to_map(position, redshift, mass, MaglimHodProps[i], afactor, (nzin*factor).astype(int), bins, nside, mask = gmask_ud)
    zout, nzout = utils.get_nz(zs, nbins = bins, density = False, getbins = False)

    extra = f"_{model}_{version}" if version else ""

    #not saving the catalog just to save space
    hp.write_map(maps_data/f"websky_map_zbin_{i}{extra}.fits", mappa, overwrite = True)
    np.savetxt(maps_data/f"nz_websky_zbin_{i}{extra}.txt", np.array([zout, nzout]).T)
    np.savetxt(maps_data/f"nz_input_zbin_{i}{extra}.txt", np.array([zin, nzin]).T)


