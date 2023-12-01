"""
Defines the HOD base class.
"""

import jax
import jax.numpy as jnp

import numpy as np

import scipy


class HOD:

    """
    With NFW profile.
    """
  
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

    
    @jax.jit
    def poisson(self, lam):
        return self._poisson(self.key, lam)
    
    @jax.jit
    def binomial(self, n, p):
        return self._binomial(self.key, n, p)
    
    @jax.jit
    def bernoulli(self, p):
        return self._bernoulli(self.key, p)
    
    @staticmethod
    @jax.jit
    def _bernoulli(key, p):
        return jax.random.bernoulli(key, p)

    @staticmethod
    @jax.jit
    def _poisson(key, lam):
        return jax.random.poisson(key, lam)
    
    @staticmethod
    @jax.jit
    def _binomial(key, n, p):
        return jax.random.binomial(key, n, p)
    
    @jax.jit
    def _draw_central_average_numbers(self, mean_centrals: jnp.ndarray, threshold_centrals: float = 1e-4) -> jnp.ndarray:
        mean_centrals = mean_centrals if threshold_centrals == 0. else jnp.clip(mean_centrals, a_min = threshold_centrals, a_max = 1.-threshold_centrals)
        drawn_central_average_numbers = self.bernoulli(mean_centrals)
        return drawn_central_average_numbers.astype(jnp.int32)
    
    def centrals(self, mean_centrals: jnp.ndarray) -> jnp.ndarray:
        return self._draw_central_average_numbers(mean_centrals)
    
    def satellites(self, mean_satellites: jnp.ndarray) -> jnp.ndarray:
        return self.poisson(mean_satellites)
    
    #@jax.jit
    def sat_positions(self, central_positions: jnp.ndarray, rvir: jnp.ndarray, concentration: jnp.ndarray = 7.) -> jnp.ndarray:
        """
        The central positions are for each central selected with a some number of satellites.
        """
        profile = self.nfw_profile((np.array(central_positions.shape)[0], ), concentration = concentration)*rvir
        delta = self.get_delta_displacement_from_profile(profile)
        return self.shift_point(central_positions, delta)


    @jax.jit
    def cen_positions(self, data) -> jnp.ndarray:
        """
        Parameters
        ----------
        data : jnp.ndarray
            The data array containing the halo positions in xyz coordinates.
        """
        return data
    
    @staticmethod
    @jax.jit
    def shift_point(x0: jnp.ndarray, delta: jnp.ndarray):
        return x0+delta
    

    @jax.jit
    def get_delta_displacement_from_profile(self, profile: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.pi*self.uniform(profile.shape)
        phi = 2*jnp.pi*self.uniform(profile.shape)
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        delta = profile*jnp.array([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta])
        return delta.T
    

    def uniform(self, shape):
        return self._uniform(self.key, shape)
    
    @staticmethod
    #@jax.jit
    def _uniform(key, shape):
        return jax.random.uniform(key, shape)
    
    #@jax.jit
    def nfw_profile(self, N, concentration: jnp.ndarray):
        uniform = self.uniform(N)
        return self._nfw_r(uniform, concentration)

    @jax.jit
    def _nfw_r(self, uniform: jnp.ndarray, concentration: jnp.ndarray):
        '''
        Inverse CDF
        See Eq. (6) from https://arxiv.org/pdf/1805.09550.pdf
        '''
        argument = -uniform*self.M_function(1, concentration)-1
        partial_result = -self.lambertw(jnp.exp(argument))
        q = -1/concentration*(1+1/partial_result)
        return q
    
    @staticmethod
    @jax.jit
    def M_function(q: jnp.ndarray, concentration: float):
        '''
        Unnormalized M function
        See Eq. (2) https://arxiv.org/pdf/1805.09550.pdf
        '''
        temp = q*concentration
        return jnp.log(1+temp)-temp/(1+temp)
    
    @staticmethod
    @jax.jit
    def lambertw(z: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(
            lambda z: scipy.special.lambertw(z).real.astype(z.dtype), z, z)












    def __call__(self, x):
        pass
    

    def populate(self, x):
        pass
    

    
    def random(self, size):
        return jnp.random.random(key = self.key, size = size)

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {"seed": self.seed}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    

from jax import tree_util
tree_util.register_pytree_node(HOD,
                               HOD._tree_flatten,
                               HOD._tree_unflatten)