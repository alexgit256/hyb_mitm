from scipy.special import betainc
import math
from scipy.special import gammaln
from math import exp

#as per Lemma 1 (https://eprint.iacr.org/2025/2195.pdf)
def projected_norm_cdf(full_dimension, full_norm, proj_dimension, proj_norm):
    """
    CDF for the projected norm when a vector of norm `full_norm` in R^full_dimension
    is projected onto a random proj_dimension-dimensional subspace.
    """
    n = full_dimension
    m = proj_dimension
    v = full_norm
    x = proj_norm

    if x <= 0:
        return 0.0
    if x >= v:
        return 1.0

    alpha = m / 2.0
    beta = (n - m) / 2.0
    z = (x / v) ** 2

    return betainc(alpha, beta, z)

"""
If Z ~ Beta(a,b) with CDF I_{x}(a,b) [also called betainc] then: f_Z(z) = \\frac{z^{a-1}(1-z)^{b-1}} / B(a,b) and E[Z^p] = \\int_{0}^1 z^p f_Z(z) dz
E[Z^p] = 1/B(a,b) \\int_{0}^{1} z^p z^{a-1} (1-z)^{b-1} dz and thus, by def. of B(a,b):
E[Z^p] = \\frac{ B(a+p,b) }{ B(a,b) } if a+p>0.

Then let X=B(a,b) -> Z = (X/v)^2 ~ Beta(a,b) by def. of projected_norm_cdf
X = u Z^{1/2} and E[X] = v E[\\sqrt{Z}] and finally
E[X] = E[\sqrt{Z}] = v \\frac{ B(a+0.5,b) }{ B(a,b) } = \\frac{ Г(a+0.5)Г(a+b) }{ Г(a)Г(a+b+0,5) } 
"""
def expected_proj_norm(full_dimension, full_norm, proj_dimension):
    """
    expected norm of the projection of a fixed vector of norm `full_norm`
    in R^full_dimension onto a random proj_dimension-dimensional subspace.
    """
    n = full_dimension
    m = proj_dimension
    v = full_norm

    if v < 0:
        raise ValueError("full_norm must be nonnegative")
    if not (0 <= m <= n):
        raise ValueError("proj_dimension must satisfy 0 <= proj_dimension <= full_dimension")

    if v == 0 or m == 0:
        return 0.0
    if m == n:
        return float(v)

    # Г(a+0.5)Г(a+b) }{ Г(a)Г(a+b+0.5) = Г( (m + 1) / 2.0)Г( n/2 ) }{ Г( m/2 )Г( (n+1)/2 )
    log_expected_factor = (
        gammaln((m + 1) / 2.0)
        + gammaln(n / 2.0)
        - gammaln(m / 2.0)
        - gammaln((n + 1) / 2.0)
    )

    return v * exp(log_expected_factor)