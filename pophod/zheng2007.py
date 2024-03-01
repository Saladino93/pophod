"""
Zheng et al. (2007) model for the mean number of centrals and satellites in a halo of mass Mh.
"""

import jax.numpy as jnp
import jax


class Zheng2007(object):
  
    def __init__(self, log10Mmin: jnp.ndarray, sigma_logM: jnp.ndarray, fcen: jnp.ndarray, log10M1: jnp.ndarray, alpha: jnp.ndarray, kappa: jnp.ndarray = 0., seed: int = 0):
        self.M1 = jnp.power(10., log10M1)
        self.log10M1 = log10M1
        self.log10Mmin = log10Mmin
        self.sigma_logM = sigma_logM
        self.fcen = fcen
        self.alpha = alpha
        self.kappa = kappa

        self.seed = seed
        self.key = jax.random.PRNGKey(seed)


    def __call__(self, Mh: jnp.ndarray):
        """
        Returns the mean number of centrals and satellites in a halo of mass Mh, as well as the total number (the sum of the former two).
        """
        ncentral = self.mean_centrals(Mh)
        nsatellite = self.mean_satellites(Mh, ncentral)
        total = ncentral + nsatellite
        return (ncentral, nsatellite, total)
    
    @jax.jit
    def mean_centrals(self, Mh: jnp.ndarray):
        """
        Get the mean number of central galaxies in a halo of mass Mh.
        """
        log10Mh = jnp.log10(Mh)
        return self._get_mean_number_center(log10Mh, self.fcen, self.log10Mmin, self.sigma_logM)
    
    @staticmethod
    @jax.jit
    def _get_mean_number_center(log10Mh: jnp.ndarray, fcen: float, log10Mmin: float, sigma_logM: float) -> jnp.ndarray:
        '''
        See for example equation (1) of https://arxiv.org/pdf/2106.08438.pdf
        '''
        return 0.5 * (1.0 + jax.scipy.special.erf((log10Mh - log10Mmin) / sigma_logM)) * fcen
    

    @jax.jit
    def mean_satellites(self, Mh: jnp.ndarray, mean_number_center: jnp.ndarray) -> jnp.ndarray:
        return self._get_mean_number_satellite(mean_number_center, Mh, self.M1, self.alpha, self.kappa, self.log10Mmin)
    
    @staticmethod
    @jax.jit
    def _get_mean_number_satellite(mean_number_center: jnp.ndarray, Mh: jnp.ndarray, M1: float, alpha: float, kappa: float = 0., Mmin: float = 0.) -> jnp.ndarray:
        '''
        See for example equation (2) of https://arxiv.org/pdf/2106.08438.pdf
        '''
        result = mean_number_center*((Mh-kappa*Mmin)/M1)**alpha*(Mh>kappa*Mmin)
        return result

    def _tree_flatten(self):
        children = (self.log10Mmin,
                    self.sigma_logM,
                    self.fcen,
                    self.log10M1,
                    self.alpha,
                    self.kappa)  # arrays / dynamic values, for now no need to dynamic values for differentiation
        aux_data = {"seed": self.seed}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    

from jax import tree_util
tree_util.register_pytree_node(Zheng2007,
                               Zheng2007._tree_flatten,
                               Zheng2007._tree_unflatten)