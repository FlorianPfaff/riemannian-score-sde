import abc

import haiku as hk
import jax.numpy as jnp
import jax

from hydra.utils import instantiate
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric, gegenbauer_polynomials
from geomstats.geometry.base import LevelSet as EmbeddedManifold
from geomstats.geometry.base import LevelSet
from geomstats.geometry.open_hemisphere import OpenHemisphere
from geomstats.backend import random, where, clip, linalg, arccos
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.backend as gs
from math import pi

# def get_exact_div_fn(fi_fn, Xi=None):
#     "flatten all but the last axis and compute the true divergence"

#     def div_fn(x: jnp.ndarray, t: float):
#         x_shape = x.shape
#         dim = np.prod(x_shape[1:])
#         t = jnp.expand_dims(t.reshape(-1), axis=-1)
#         x = jnp.expand_dims(x, 1)  # NOTE: need leading batch dim after vmap
#         t = jnp.expand_dims(t, 1)
#         jac = jax.vmap(jax.jacrev(fi_fn, argnums=0))(x, t)
#         jac = jac.reshape([x_shape[0], dim, dim])
#         if Xi is not None:
#             jac = jnp.einsum("...nd,...dm->...nm", jac, Xi)
#         div = jnp.trace(jac, axis1=-1, axis2=-2)
#         return div

#     return div_fn


class VectorFieldGenerator(hk.Module, abc.ABC):
    def __init__(self, architecture, embedding, output_shape, manifold):
        """X = fi * Xi with fi weights and Xi generators"""
        super().__init__()
        self.net = instantiate(architecture, output_shape=output_shape)
        self.embedding = instantiate(embedding, manifold=manifold)
        self.manifold = manifold

    @staticmethod
    @abc.abstractmethod
    def output_shape(manifold):
        """Cardinality of the generating set."""

    def _weights(self, x, t):
        """shape=[..., card=n]"""
        return self.net(*self.embedding(x, t))

    @abc.abstractmethod
    def _generators(self, x):
        """Set of generating vector fields: shape=[..., d, card=n]"""

    @property
    def decomposition(self):
        return lambda x, t: self._weights(x, t), lambda x: self._generators(x)

    def __call__(self, x, t):
        fi_fn, Xi_fn = self.decomposition
        fi, Xi = fi_fn(x, t), Xi_fn(x)
        out = jnp.einsum("...n,...dn->...d", fi, Xi)
        # NOTE: seems that extra projection is required for generator=eigen
        # during the ODE solve cf tests/test_lkelihood.py
        out = self.manifold.to_tangent(out, x)
        return out

    def div_generators(self, x):
        """Divergence of the generating vector fields: shape=[..., card=n]"""


class DivFreeGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        return manifold.isom_group.dim

    def _generators(self, x):
        return self.manifold.div_free_generators(x)

    def div_generators(self, x):
        shape = [*x.shape[:-1], self.output_shape(self.manifold)]
        return jnp.zeros(shape)


class OpenHemisphereWithWraparound(OpenHemisphere):
    def _single_point_div_free_generators(self, x):
        dim = x.shape[-1] - 1
        generators = []
        for i in range(dim):
            for j in range(i + 1, dim + 1):
                generator = jnp.zeros_like(x)
                generator = generator.at[i].set(x[j])
                generator = generator.at[j].set(-x[i])
                generators.append(generator)
        return jnp.stack(generators, axis=-1)
    
    def _log_heat_kernel(self, x0, x, t, n_max):
        if len(t.shape) == len(x.shape):
            t = t[..., 0]
        t = t / 2  # NOTE: to match random walk
        d = self.dim

        cos_theta = gs.sum(x0 * x, axis=-1)
        cos_theta = gs.clip(cos_theta, -1.0, 1.0)  # Ensure valid input for acos
        theta = gs.arccos(cos_theta)

        # Since hyperhemisphere:
        # Ensure handling of both point and antipodal point
        antipodal_theta = gs.arccos(-cos_theta)
        theta = gs.minimum(theta, antipodal_theta)

        if d == 1:
            n = gs.expand_dims(gs.arange(-n_max, n_max + 1), axis=-1)
            t = gs.expand_dims(t, axis=0)
            sigma_squared = t
            coeffs = gs.exp(-gs.power(theta + 2 * pi * n, 2) / 2 / sigma_squared)
            prob = gs.sum(coeffs, axis=0)
            prob = prob / gs.sqrt(2 * pi * sigma_squared[0])
        else:
            n = gs.expand_dims(gs.arange(0, n_max + 1), axis=-1)
            t = gs.expand_dims(t, axis=0)
            coeffs = (
                gs.exp(-n * (n + 1) * t)
                * (2 * n + d - 1)
                / (d - 1)
                / (Hypersphere(self.dim).volume/2)
            )
            P_n = gegenbauer_polynomials(
                alpha=(self.dim - 1) / 2, l_max=n_max, x=cos_theta
            )
            prob = gs.sum(coeffs * P_n, axis=0)

        return gs.log(prob)

    
    @staticmethod
    def default_metric():
        return HyperhemisphericalWraparoundMetric
    
    def random_walk(self, rng, x, t):
        next_points = Hypersphere.random_walk(self, rng, x, t)
        if not next_points:
            return None
        mask = (next_points[:, -1] < 0)[:, None]  # Shape becomes (batch_dim, 1)
        # Mirror those on lower hemisphere
        next_points = where(mask, -next_points, next_points)
        return next_points
    
    def grad_marginal_log_prob(self, x0, x, t, thresh, n_max):
        # Should be ok because the functions that it builds upon are adjusted
        return LevelSet.grad_marginal_log_prob(self, x0, x, t, thresh, n_max)
    
    def grad_log_heat_kernel_exp(self, x0, x, t):
        # Should be ok because the functions that it builds upon are adjusted
        return LevelSet.grad_log_heat_kernel_exp(self, x0, x, t)

    def div_free_generators(self, x):
        """
        Compute divergence-free vector fields on the hemisphere without worrying
        about boundary conditions because the probability of reaching the boundary is zero.

        Parameters
        ----------
        x : array-like, shape=[..., dim+1]
            Points strictly inside the hemisphere, not including the boundary.

        Returns
        -------
        generators : array-like
            Divergence-free vector fields on the hemisphere,
            shape=[..., dim+1, number of fields].
        """
        # Apply the generator function to each point in the batch
        batched_generator = jax.vmap(self._single_point_div_free_generators, in_axes=0, out_axes=0)
        return batched_generator(x)
    
    def random_normal_tangent(self, state, base_point, n_samples=1):
        """
        Generate random tangent vectors on the hemisphere.

        Parameters
        ----------
        state : PRNGKey
            JAX PRNG key.
        base_point : array-like, shape=[..., dim+1]
            Base points on the hemisphere.
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim+1]
            Random tangent vectors on the hemisphere.
        """
        mirror_flag = jax.random.bernoulli(state, shape=(base_point.shape[0],), p=0.5)
        state, ambiant_noise = random.normal(
            state=state, size=(n_samples, base_point.shape[-1])
        )
        base_points_half_mirrored = where(mirror_flag[:, None], -base_point, base_point)
        return state, self.to_tangent(vector=ambiant_noise, base_point=base_points_half_mirrored)


class HyperhemisphericalWraparoundMetric(RiemannianMetric):
    """Class for the Hyperhemispherical Metric with antipodal symmetry."""

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point, respecting antipodal symmetry.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point on the hypersphere.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        return self._space.embedding_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

    def exp(self, tangent_vec, base_point):
        exp_points = HypersphereMetric.exp(self, tangent_vec, base_point)
        # Reflect if the exp_point is outside the intended hyperhemisphere
        mask = (exp_points[:, -1] < 0)[:, None]  # Shape becomes (16384, 1)

        exp_points = where(mask, -exp_points, exp_points)
        return exp_points

    def log(self, point, base_point):
        # Determine if point or -point is closer to base_point
        direct_logs = HypersphereMetric.log(self, point, base_point)
        antipodal_logs = HypersphereMetric.log(self, -point, base_point)
        
        # Compute norms (distances) for direct and antipodal logs
        direct_distances = linalg.norm(direct_logs, axis=1)
        antipodal_distances = linalg.norm(antipodal_logs, axis=1)
        
        # Compare distances and choose the shorter for each pair in the batch
        mask = antipodal_distances < direct_distances
        result_logs = where(mask[:, None], antipodal_logs, direct_logs)
        return result_logs

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points, considering symmetry.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            First point on the hypersphere.
        point_b : array-like, shape=[..., dim + 1]
            Second point on the hypersphere.

        Returns
        -------
        dist : array-like, shape=[..., 1]
            Geodesic distance between the two points.
        """
        inner_prod = point_a @ point_b
        if inner_prod < 0:
            inner_prod = -inner_prod
        cos_angle = clip(inner_prod / (linalg.norm(point_a) * linalg.norm(point_b)), -1, 1)
        return arccos(cos_angle)

    def squared_dist(self, point_a, point_b):
        """Squared geodesic distance between two points, considering symmetry.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point on the hypersphere.
        point_b : array-like, shape=[..., dim]
            Point on the hypersphere.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
        """
        return self.dist(point_a, point_b) ** 2


class EigenGenerator(VectorFieldGenerator):
    """Gradient of laplacien eigenfunctions with eigenvalue=1"""

    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)
        assert isinstance(manifold, Hypersphere)

    @staticmethod
    def output_shape(manifold):
        return manifold.embedding_space.dim

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def div_generators(self, x):
        # NOTE: Empirically need this factor 2 to match AmbientGenerator but why??
        return -self.manifold.dim * 2 * x


class AmbientGenerator(VectorFieldGenerator):
    """Equivalent to EigenGenerator"""

    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        if isinstance(manifold, EmbeddedManifold) or isinstance(manifold, LevelSet):
            output_shape = manifold.embedding_space.dim
        else:
            output_shape = manifold.dim
        return output_shape

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def __call__(self, x, t):
        # `to_tangent`` have an 1/sq_norm(x) term that wrongs the div
        return self.manifold.to_tangent(self.net(x, t), x)


class LieAlgebraGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return self.manifold.lie_algebra.basis

    def __call__(self, x, t):
        x = x.reshape((x.shape[0], self.manifold.dim, self.manifold.dim))
        fi_fn, Xi_fn = self.decomposition
        x_input = x.reshape((*x.shape[:-2], -1))
        fi, Xi = fi_fn(x_input, t), Xi_fn(x)
        out = jnp.einsum("...i,ijk ->...jk", fi, Xi)
        out = self.manifold.compose(x, out)
        # out = self.manifold.to_tangent(out, x)
        return out.reshape((x.shape[0], -1))


class TorusGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

        self.rot_mat = jnp.array([[0, -1], [1, 0]])

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return (
            self.rot_mat @ x.reshape((*x.shape[:-1], self.manifold.dim, 2))[..., None]
        )[..., 0]

    def __call__(self, x, t):
        weights_fn, fields_fn = self.decomposition
        weights = weights_fn(x, t)
        fields = fields_fn(x)

        return (fields * weights[..., None]).reshape(
            (*x.shape[:-1], self.manifold.dim * 2)
        )


class CanonicalGenerator:
    def __init__(self, architecture, embedding, output_shape=None, manifold=None):
        self.net = instantiate(architecture, output_shape=output_shape)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def __call__(self, x, t):
        return self.net(x, t)


class ParallelTransportGenerator:
    def __init__(self, architecture, embedding, output_shape=None, manifold=None):
        self.net = instantiate(architecture, output_shape=output_shape)
        self.manifold = manifold

    @staticmethod
    def output_shape(manifold):
        # return manifold.dim
        return manifold.identity.shape[-1]

    def __call__(self, x, t):
        """
        Rescale since ||s(x, t)||^2_x = s(x, t)^t G(x) s(x, t) = \lambda(x)^2 ||s(x, t)||^2_2
        with G(x)=\lambda(x)^2 Id
        """
        tangent = self.net(x, t)
        tangent = self.manifold.metric.transpfrom0(x, tangent)
        return tangent
