from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from riemannian_score_sde.models.distribution import (
    WrapNormDistribution as WrappedNormal,
)
from pyrecest.distributions import VonMisesFisherDistribution, HypersphericalMixture
from geomstats.backend import array, random, ones_like, prod

class vMFMixture:
    def __init__(
        self, batch_dims, rng, manifold, mu, kappa, weights=[0.5, 0.5], **kwargs
    ):
        if len(mu) != len(kappa):
            raise ValueError("The length of mu_array and kappa_values must be the same.")

        self.manifold = manifold
        self.mu = array(mu)
        vmfs = [VonMisesFisherDistribution(array(mu_curr), kappa_curr) for mu_curr, kappa_curr in zip(mu, kappa)]
        self.mixture = HypersphericalMixture(vmfs, weights)
        
        self.batch_dims = batch_dims
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.mixture.sample(prod(array(self.batch_dims)))
        return (samples, None)


class WrapNormMixtureDistribution:
    def __init__(
        self,
        batch_dims,
        manifold,
        mean,
        scale,
        seed=0,
        rng=None,
    ):
        self.mean = array(mean)
        self.K = self.mean.shape[0]
        self.scale = array(scale)
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.rng = rng if rng is not None else jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        n_samples = np.prod(self.batch_dims)
        ks = jnp.arange(self.K)
        self.rng, next_rng = jax.random.split(self.rng)
        _, k = random.choice(state=next_rng, a=ks, n=n_samples)
        mean = self.mean[k]
        scale = self.scale[k]
        tangent_vec = self.manifold.random_normal_tangent(
            next_rng, self.manifold.identity, n_samples
        )[1]
        tangent_vec *= scale
        tangent_vec = self.manifold.metric.transpfrom0(mean, tangent_vec)
        samples = self.manifold.metric.exp(tangent_vec, mean)
        return (samples, None)

    def log_prob(self, x):
        def component_log_prob(mean, scale):
            return WrappedNormal(self.manifold, scale, mean).log_prob(x)

        component_log_like = jax.vmap(component_log_prob)(self.mean, self.scale)
        b = 1 / self.K * ones_like(component_log_like)
        return logsumexp(component_log_like, axis=0, b=b)
