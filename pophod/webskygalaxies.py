"""
Script to get galaxy maps from catalogs, using an HOD model augumented with a redshift weighting.
"""

from pophod import hod, zheng2007
from pophod.utils import websky
import pyccl
import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp



class WebSkyGalaxyCatalog(object):
    """
    Class to generate galaxy maps from halo catalogs, using an HOD model augmented with a redshift weighting.

    Specialised for Websky simulations, but in principle can be used with any halo catalog with similar inputs.
    """

    def __init__(self, seed: int, cosmology: dict):
        self.seed = seed
        self.HOD = hod.HOD(seed = seed)
        #hardcoded for now the value of mnu
        mnu = 0
        del cosmology["Omega_M"], cosmology["Omega_L"]
        get_cosmo = lambda Omega_C, Omega_B, h, sigma_8, n_s: pyccl.Cosmology(Omega_c = Omega_C, Omega_b = Omega_B, h = h, sigma8=sigma_8, n_s=n_s, m_nu=mnu)
        self.cosmo = get_cosmo(**cosmology)


    def get_galaxy_catalog(self, input_halos: np.ndarray, input_redshifts: np.ndarray, mass: np.ndarray, props: dict, afactor: float = 1.0):
        Zheng = zheng2007.Zheng2007(**props)
        centrals, satellites, totals = Zheng(mass)
        central_galaxies = self.HOD.centrals(centrals) #which galaxies are centrals
        satellite_galaxies = self.HOD.satellites(satellites) #which galaxies are satellites

        selection = central_galaxies.astype(bool)
        position_central = input_halos[selection]
        mass_central = mass[selection]
        redshift_central = input_redshifts[selection]
        satellite_centeral = satellite_galaxies[selection]

        #should not be that relevant the exact concentration model
        conc = self.Bhatt(self.cosmo, mass_central, 1/(1+redshift_central))

        rho_m0 = 2.775e11 * (self.cosmo["Omega_c"]+self.cosmo["Omega_b"]) * self.cosmo["h"]**2
        rho_mz = rho_m0 * jnp.power(1+redshift_central, 3)
        R_central = jnp.power(3*mass_central/(4*jnp.pi*200*rho_mz), 1/3) # Mpc

        #this operations are for broadcasting the values of the centrals to accommodate the satellites
        central_position_for_sat = jnp.repeat(position_central, satellite_centeral, axis = 0)
        R_for_sat = jnp.repeat(R_central, satellite_centeral, axis = 0)
        conc_for_sat = jnp.repeat(conc, satellite_centeral, axis = 0)

        satellite_positions = self.HOD.sat_positions(central_position_for_sat, R_for_sat, concentration = conc_for_sat*afactor*np.ones_like(R_for_sat))

        #shuffle indices
        #position_central = position_central[jax.random.permutation(jax.random.PRNGKey(42), position_central.shape[0])]
        #satellite_positions = satellite_positions[jax.random.permutation(jax.random.PRNGKey(42), satellite_positions.shape[0])]
        #print("Finished shuffling")

        #galaxy catalog is the concatenation of central and satellite positions
        gal_catalog = jnp.concatenate([position_central, satellite_positions], axis = 0)

        #obtain the chi to get redshifts
        chi = jnp.sqrt(jnp.sum(gal_catalog**2., axis = 1))#Mpc
        redshift_catalog = 1/self.cosmo.scale_factor_of_chi(chi)-1

        return gal_catalog, redshift_catalog

    @staticmethod
    def Bhatt(cosmo, M, a):
        #Bhattacharya et al. 2013, see table in paper
        Dz = cosmo.growth_factor(a)
        nu = (1.12*(M/5e13)**0.3 + 0.53)/Dz
        A, B, C = 0.9, 7.7, -0.29
        result = Dz**A*B*nu**C
        return result


    @staticmethod
    def select(mass, position, redshift, selecting_variable, vmin, vmax):
        """
        Utility function to select halos in a given redshift range. In principle you can use whatever variable you want to select the halos.
        """
        mask = (selecting_variable >= vmin) & (selecting_variable <= vmax)
        return mass[mask], position[mask], redshift[mask]
    
    @staticmethod
    def reweight_halos_bins(input_positions: np.ndarray, input_redshifts: np.ndarray, outnz: np.ndarray, outbins: np.ndarray):
        
        #basically doing an histogram with the indices, with outbins already sorted
        #note, for a v value in input_redshifts, left will give the index i in outbins
        #such that a[i-1] < v <= a[i], so I will have to subtract 1 to get the correct index
        indices = np.searchsorted(outbins, input_redshifts, side = 'left')-1

        # Use boolean indexing with flatnonzero and random.choice to select samples
        selected = np.concatenate([np.random.choice(z_sel, size = nz, replace = len(z_sel) < nz) for nz, z_sel in zip(outnz, (np.flatnonzero(indices == i) for i in range(len(outbins) - 1))) if len(z_sel) > 0])
        #count effective number of galaxies
        total = sum([len(z_sel) for z_sel in (np.flatnonzero(indices == i) for i in range(len(outbins) - 1)) if len(z_sel) > 0])
        print("Total galaxies selected: ", total)

        new_zs = input_redshifts[selected]
        new_position = input_positions[selected]
        return new_position, new_zs
    
    @staticmethod
    def get_map(gal_catalog, nside):
        return websky.catalogue_to_map(gal_catalog, nside = nside)
    
    @staticmethod
    def masking(mask, gal_catalog, redshift_catalog):
        #there we would like to get the galaxies inside the mask
        nside = hp.get_nside(mask)
        #first, we get the indices where the mask>0.
        unmasked_pixels = np.where(mask == True)[0]
        #then, all of the indices of the input catalog, so that we are in the same index space 
        pix = hp.vec2pix(nside, gal_catalog[:, 0], gal_catalog[:, 1], gal_catalog[:, 2])
        #finally, we get the indices inside the mask 
        indices_in_mask = np.nonzero(np.isin(pix, unmasked_pixels))[0]
        return gal_catalog[indices_in_mask], redshift_catalog[indices_in_mask]

    def from_cat_to_map(self, input_halos, input_redshifts, mass, props, afactor, outnz, outbins, nside, mask = None):
        gal_catalog, redshift_catalog = self.get_galaxy_catalog(input_halos, input_redshifts, mass, props, afactor)
        #mask operation could also be moved inside get_galaxy_catalog, for more efficiency
        #you check halos inside the mask, and then you get the galaxies, instead of what I am doing now
        #now, if mask is not None, we get all objects within the mask
        if mask is not None:
            print("Masking...")
            gal_catalog, redshift_catalog = self.masking(mask, gal_catalog, redshift_catalog)

        #reweighted_gal_catalog, reweighted_redshift_catalog = gal_catalog, redshift_catalog #
        reweighted_gal_catalog, reweighted_redshift_catalog = self.reweight_halos_bins(gal_catalog, redshift_catalog, outnz = outnz, outbins = outbins)
        return self.get_map(reweighted_gal_catalog, nside), reweighted_redshift_catalog