"""Classical kernels on the same scaled inputs as the quantum kernels."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _sq(X, Y, ls=1.0):
    X, Y = np.atleast_2d(X) / ls, np.atleast_2d(Y) / ls
    return cdist(X, Y, "sqeuclidean")


def rbf(X, Y=None, lengthscale=1.0):
    """lengthscale may be scalar or per-dimension (ARD)."""
    Y = X if Y is None else Y
    return np.exp(-0.5 * _sq(X, Y, np.asarray(lengthscale)))


def matern52(X, Y=None, lengthscale=1.0):
    Y = X if Y is None else Y
    r = np.sqrt(np.clip(_sq(X, Y, np.asarray(lengthscale)), 0, None))
    s5 = np.sqrt(5.0) * r
    return (1 + s5 + 5.0 / 3.0 * r ** 2) * np.exp(-s5)


def product_cos2(X, Y=None, bandwidth=1.0):
    """Exact closed form of the product-angle fidelity kernel: prod_j cos^2(bw*pi*(x_j - y_j)/2)."""
    Y = X if Y is None else Y
    D = np.atleast_2d(X)[:, None, :] - np.atleast_2d(Y)[None, :, :]
    return np.prod(np.cos(bandwidth * np.pi * D / 2) ** 2, axis=-1)


def linear(X, Y=None):
    Y = X if Y is None else Y
    return np.atleast_2d(X) @ np.atleast_2d(Y).T
