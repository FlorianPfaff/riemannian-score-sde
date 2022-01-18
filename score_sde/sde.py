"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Modified code from https://github.com/yang-song/score_sde
"""
from abc import ABC, abstractmethod
from typing import Tuple, Callable

import jax
import numpy as np
import jax.numpy as jnp
from geomstats.geometry.euclidean import Euclidean

# from .utils import batch_mul
def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


# class SDE(abc.ABC):
#     """SDE abstract class. Functions are designed for a mini-batch of inputs."""

#     def __init__(self, N: int):
#         """Construct an SDE.

#         Args:
#           N: number of discretization time steps.
#         """
#         super().__init__()
#         self.N = N

#     @property
#     @abc.abstractmethod
#     def T(self) -> float:
#         """End time of the SDE."""
#         pass

#     @abc.abstractmethod
#     def sde(self, x: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Return the drift and diffusion coefficients for the SDE"""
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def marginal_prob(
#         self, x: jnp.ndarray, t: float
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Parameters (mean, std) to determine the marginal distribution of the SDE, $p_t(x)$."""
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def prior_sampling(
#         self, rng: jax.random.KeyArray, shape: jnp.ndarray
#     ) -> jnp.ndarray:
#         """
#         Generate sample from the prior distribution, $p_T(x)$.
#         Shape gives the expected shape of a single sample
#         """
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def prior_logp(self, z: jnp.ndarray) -> jnp.ndarray:
#         """Compute log-density of the prior distribution.

#         Useful for computing the log-likelihood via probability flow ODE.

#         Args:
#           z: latent code
#         Returns:
#           log probability density
#         """
#         raise NotImplementedError()

#     def discretize(self, x: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

#         Useful for reverse diffusion sampling and probabiliy flow sampling.
#         Defaults to Euler-Maruyama discretization. TODO: Note sure what they mean by this?

#         Args:
#           x: a JAX tensor.
#           t: a JAX float representing the time step (from 0 to `self.T`)

#         Returns:
#           f, G
#         """
#         dt = 1 / self.N
#         drift, diffusion = self.sde(x, t)
#         f = drift * dt
#         G = diffusion * jnp.sqrt(dt)
#         return f, G

#     def reverse(
#         self,
#         score_fn: Callable[[jnp.ndarray], jnp.ndarray],
#         probability_flow: bool = False,
#     ):
#         """Create the reverse-time SDE/ODE.

#         Args:
#           score_fn: A time-dependent score-based model that takes x and t and returns the score.
#           probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
#         """
#         N = self.N
#         T = self.T
#         sde_fn = self.sde
#         discretize_fn = self.discretize

#         # Build the class for reverse-time SDE.
#         class RSDE(self.__class__):
#             def __init__(self):
#                 self.N = N
#                 self.probability_flow = probability_flow

#             @property
#             def T(self):
#                 return T

#             def sde(self, x, t):
#                 """Create the drift and diffusion functions for the reverse SDE/ODE."""
#                 drift, diffusion = sde_fn(x, t)
#                 score = score_fn(x, t)
#                 drift = drift - batch_mul(
#                     diffusion ** 2, score * (0.5 if self.probability_flow else 1.0)
#                 )
#                 # Set the diffusion function to zero for ODEs.
#                 diffusion = (
#                     jnp.zeros_like(diffusion) if self.probability_flow else diffusion
#                 )
#                 return drift, diffusion

#             def discretize(self, x, t):
#                 """Create discretized iteration rules for the reverse diffusion sampler."""
#                 f, G = discretize_fn(x, t)
#                 rev_f = f - batch_mul(
#                     G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.0)
#                 )
#                 rev_G = jnp.zeros_like(G) if self.probability_flow else G
#                 return rev_f, rev_G

#         return RSDE()


class SDE(ABC):
    def __init__(self, tf: float = 1, t0: float = 0):
        """Abstract definition of an SDE"""

        self.t0 = t0
        self.tf = tf

        # if ((N == None) and (dt == None)) or ((N != None) and (dt != None)):
        #     raise ValueError("Exactly one of dt and N must not be None")

        # if N != None:
        #     self.N = N
        #     self.dt = (self.tf - self.t0) / self.N

        # if dt != None:
        #     self.N = int((self.tf - self.t0) / dt)
        #     self.dt = (self.tf - self.t0) / self.N

    @abstractmethod
    def coefficients(self, x, t):
        """Compute the drift and diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        pass

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        raise NotImplementedError()

    def marginal_log_prob(self, x0, x, t):
        """Compute the log marginal distribution of the SDE, $log p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x0: jnp.ndarray
            Location of the start of the diffusion
        x : jnp.ndarray
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        raise NotImplementedError()

    def grad_marginal_log_prob(self, x0, x, t):
        """Compute the log marginal distribution and its gradient

        Parameters
        ----------
        x0: jnp.ndarray
            Location of the start of the diffusion
        x : jnp.ndarray
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        logp_grad_fn = jax.value_and_grad(
            self.marginal_log_prob, argnums=1, has_aux=False
        )
        logp, logp_grad = jax.vmap(logp_grad_fn)(x0, x, t)
        logp_grad = self.manifold.to_tangent(logp_grad, x)
        return logp, logp_grad

    def sample_limiting_distribution(self, rng, shape):
        """Generate samples from the limiting distribution, $p_{t_f}(x)$.
        (distribution may not exist / be inexact)

        Parameters
        ----------
        rng : jnp.random.KeyArray
        shape : Tuple
            Shape of the samples to sample.
        """
        raise NotImplementedError()

    def limiting_distribution_logp(self, z):
        """Compute log-density of the limiting distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: limiting distribution sample
        Returns:
          log probability density
        """
        raise NotImplementedError()

    def discretize(self, x, t, dt):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at

        Returns:
            f, G - the discretised SDE drift and diffusion coefficients
        """
        drift, diffusion = self.coefficients(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(jnp.abs(dt))
        return f, G

    def reverse(self, score_fn):
        return RSDE(self, score_fn)

    def probability_ode(self, score_fn):
        return ProbabilityFlowODE(self, score_fn)


class ProbabilityFlowODE:
    def __init__(self, sde: SDE, score_fn=None):
        self.sde = sde

        self.t0 = sde.t0
        self.tf = sde.tf

        if score_fn is None and not isinstance(sde, RSDE):
            raise ValueError(
                "Score function must be not None or SDE must be a reversed SDE"
            )
        elif score_fn is not None:
            self.score_fn = score_fn
        elif isinstance(sde, RSDE):
            self.score_fn = sde.score_fn

    def coefficients(self, x, t):
        drift, diffusion = self.sde.coefficients(x, t)
        score_fn = self.score_fn(x, t)
        # compute G G^T score_fn
        if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
            # if square matrix diffusion coeffs
            ode_drift = drift - 0.5 * jnp.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            ode_drift = drift - 0.5 * jnp.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        return ode_drift, jnp.zeros(drift.shape[:-1])


class RSDE(SDE):
    """Reverse time SDE, assuming the drift coefficient is spatially homogenous"""

    def __init__(self, sde: SDE, score_fn):
        super().__init__(tf=sde.t0, t0=sde.tf)

        self.sde = sde
        self.score_fn = score_fn

    def coefficients(self, x, t):
        forward_drift, diffusion = self.sde.coefficients(x, t)
        score_fn = self.score_fn(x, t)

        # compute G G^T score_fn
        if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
            # if square matrix diffusion coeffs
            reverse_drift = forward_drift - jnp.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            reverse_drift = forward_drift - jnp.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        return reverse_drift, diffusion

    def reverse(self):
        return self.sde


class VPSDE(SDE):
    def __init__(self, tf: float, t0: float = 0, beta_0=0.1, beta_f=20):
        super().__init__(tf, t0)

        self.beta_0 = beta_0
        self.beta_f = beta_f

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def coefficients(self, x, t):
        beta_t = self.beta_t(t)
        drift = -0.5 * beta_t[..., None] * x
        diffusion = jnp.sqrt(beta_t) * jnp.ones(x.shape[:-1])

        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_f - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = jnp.exp(log_mean_coeff)[..., None] * x
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def sample_limiting_distribution(self, rng, shape):
        # TODO: rename from `prior_sampling`
        return jax.random.normal(rng, shape)

    def limiting_distribution_logp(self, z):
        # TODO: rename from `prior_logp`
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.0
        return jax.vmap(logp_fn)(z)

    # def ddpm_discretize(self, x, t):
    #     """DDPM discretization."""
    #     timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
    #     beta = self.discrete_betas[timestep]
    #     alpha = self.alphas[timestep]
    #     sqrt_beta = jnp.sqrt(beta)
    #     f = batch_mul(jnp.sqrt(alpha), x) - x
    #     G = sqrt_beta
    #     return f, G


class subVPSDE(SDE):
    def __init__(self, beta_0=0.1, beta_f=20):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_0: value of beta(t_0)
          beta_f: value of beta(t_f)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_0
        self.beta_f = beta_f

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def coefficients(self, x, t):
        beta_t = self.beta_t(t)
        drift = -0.5 * beta_t[..., None] * x
        discount = 1.0 - jnp.exp(
            -2 * self.beta_0 * t - (self.beta_f - self.beta_0) * t ** 2
        )
        diffusion = jnp.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_f - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = jnp.exp(log_mean_coeff)[..., None] * x
        std = 1 - jnp.exp(2.0 * log_mean_coeff)
        return mean, std

    def sample_limiting_distribution(self, rng, shape):
        # TODO: rename from `prior_sampling`
        return jax.random.normal(rng, shape)

    def limiting_distribution_logp(self, z):
        # TODO: rename from `prior_logp`
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.0
        return jax.vmap(logp_fn)(z)


class VESDE(SDE):
    pass


#     def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
#         """Construct a Variance Exploding SDE.

#         Args:
#           sigma_min: smallest sigma.
#           sigma_max: largest sigma.
#           N: number of discretization steps
#         """
#         super().__init__(N)
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.discrete_sigmas = jnp.exp(
#             np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
#         )
#         self.N = N

#     @property
#     def T(self):
#         return 1

#     def sde(self, x, t):
#         sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
#         drift = jnp.zeros_like(x)
#         diffusion = sigma * jnp.sqrt(
#             2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min))
#         )
#         return drift, diffusion

#     def marginal_prob(self, x, t):
#         std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
#         mean = x
#         return mean, std

#     def prior_sampling(self, rng, shape):
#         return jax.random.normal(rng, shape) * self.sigma_max

#     def prior_logp(self, z):
#         shape = z.shape
#         N = np.prod(shape[1:])
#         logp_fn = lambda z: -N / 2.0 * jnp.log(
#             2 * np.pi * self.sigma_max ** 2
#         ) - jnp.sum(z ** 2) / (2 * self.sigma_max ** 2)
#         return jax.vmap(logp_fn)(z)

#     def discretize(self, x, t):
#         """SMLD(NCSN) discretization."""
#         timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
#         sigma = self.discrete_sigmas[timestep]
#         adjacent_sigma = jnp.where(
#             timestep == 0, jnp.zeros_like(timestep), self.discrete_sigmas[timestep - 1]
#         )
#         f = jnp.zeros_like(x)
#         G = jnp.sqrt(sigma ** 2 - adjacent_sigma ** 2)
#         return f, G
