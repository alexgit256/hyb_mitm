from random import randrange, choices, shuffle, seed as set_seed
import numpy as np
import json

from math import ceil

def binomial_dist(eta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # centered binomial with parameter eta:
    # sum of 2*eta fair bits minus eta
    return rng.binomial(2 * eta, 0.5) - eta


def binomial_vec(n, eta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.binomial(2 * eta, 0.5, size=n) - eta


def ternary_vec(n, w, rng=None):
    """
    For 0 <= w <= 1/2, each coordinate is:
      1 with prob w,
     -1 with prob w,
      0 with prob 1-2w.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0 <= w <= 0.5):
        raise ValueError("ternary parameter w must satisfy 0 <= w <= 1/2")
    return rng.choice([1, -1, 0], size=n, p=[w, w, 1 - 2 * w])


def ternary_sparse_vec(n, k, rng=None):
    """
    Exactly k nonzero entries, each independently ±1.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0 <= k <= n):
        raise ValueError("ternary_sparse parameter k must satisfy 0 <= k <= n")
    v = np.zeros(n, dtype=int)
    idx = rng.choice(n, size=k, replace=False)
    v[idx] = rng.choice([-1, 1], size=k)
    return v

def binary_vec(n, w, rng=None):
    """
    For 0 <= w <= 1, each coordinate is:
      1 with prob w,
      0 with prob 1-w.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0 <= w <= 1.0):
        raise ValueError("ternary parameter w must satisfy 0 <= w <= 1/2")
    return rng.choice([1, 0], size=n, p=[w, 1 - w])


def uniform_vec(n, a, b, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(a, b, size=n)


def continuous_gaussian_vec(n, sigma, rng=None):
    """
    Samples from N(0, sigma^2)^n, returns float vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        raise ValueError("continuous Gaussian sigma must be > 0")
    return rng.normal(loc=0.0, scale=sigma, size=n)


def discrete_gaussian_vec(n, sigma, rng=None, tailcut=8):
    """
    Samples from a truncated discrete Gaussian over Z:
        P(X = k) proportional to exp(-k^2 / (2 sigma^2))
    for k in [-B, ..., B], where B = ceil(tailcut * sigma).

    This is not the infinite-support exact sampler, but for tailcut=8
    the truncation error is tiny in most practical cases.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        raise ValueError("discrete Gaussian sigma must be > 0")

    B = max(1, ceil(tailcut * sigma))
    xs = np.arange(-B, B + 1)
    ws = np.exp(-(xs.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    ws /= ws.sum()
    return rng.choice(xs, size=n, p=ws)


def _sample_vec(dist, dim, param, rng=None):
    """
    dist: string
    param: distribution parameter
    """
    if rng is None:
        rng = np.random.default_rng()

    key = dist.lower() if isinstance(dist, str) else dist

    if key == "binomial":
        return binomial_vec(dim, param, rng=rng)

    if key == "ternary":
        return ternary_vec(dim, param, rng=rng)

    if key == "ternary_sparse":
        return ternary_sparse_vec(dim, param, rng=rng)
    
    if key == "binary":
        return binary_vec(dim, param, rng=rng)

    if key in ("gaussian", "continuous_gaussian", "normal"):
        return continuous_gaussian_vec(dim, param, rng=rng)

    if key in ("discrete_gaussian", "dgauss"):
        return discrete_gaussian_vec(dim, param, rng=rng)

    raise NotImplementedError(f"Distribution {dist!r} not implemented.")


def generateLWEInstances(
    n,
    m,
    q,
    dist_s,
    dist_param_s,
    dist_e=None,
    dist_param_e=None,
    ntar=10,
    seed=None,
):
    """
    Returns A, q, bse where:
      A   : (n x m) matrix over Z_q
      bse : list of tuples (b, s, e)
            with b = s A + e mod q

    Notes:
    - s has length n
    - e has length m
    - if dist_e/param_e omitted, uses same as secret
    - Gaussian secret/error may be float (continuous Gaussian) or int (discrete Gaussian)
    """
    if dist_e is None:
        dist_e = dist_s
    if dist_param_e is None:
        dist_param_e = dist_param_s

    rng = np.random.default_rng(seed)

    A = rng.integers(0, q, size=(n, m), dtype=np.int64)-q//2-1

    bse = []
    for _ in range(ntar):
        s = _sample_vec(dist_s, n, dist_param_s, rng=rng)
        e = _sample_vec(dist_e, m, dist_param_e, rng=rng)
        b = (s @ A + e) % q
        bse.append((b, s, e))

    return A, q, bse


def rotMatrix(poly, cyclotomic=False):
    n = len(poly)
    A = np.array([[0] * n for _ in range(n)])

    for i in range(n):
        for j in range(n):
            c = 1
            if cyclotomic and j < i:
                c = -1
            A[i][j] = c * poly[(j - i) % n]

    return A


def module(polys, rows, cols):
    if rows * cols != len(polys):
        raise ValueError("len(polys) has to equal rows*cols.")

    n = len(polys[0])
    for poly in polys:
        if len(poly) != n:
            raise ValueError("polys must not contain polynomials of varying degrees.")

    blocks = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(rotMatrix(polys[i * cols + j], cyclotomic=True))
        blocks.append(row)

    return np.block(blocks)


def binomialLWEGen(n, m, q, eta, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(n, m), dtype=np.int64)
    s = binomial_vec(n, eta, rng=rng)
    e = binomial_vec(m, eta, rng=rng)
    return A, s, e