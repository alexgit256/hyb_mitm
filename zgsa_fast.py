#Largely taken from the Leaky LWE estimator https://github.com/lducas/leaky-LWE-Estimator
#as well as the Lattice Estimator


from math import pi, exp, log, sqrt, erf
from scipy import integrate
from scipy.special import beta
from math import ceil
import numpy as np

from fpylll.util import gaussian_heuristic

def GH_sv_factor_squared(k):
    return ((pi * k)**(1. / k) * k / (2. * pi * exp(1)))

def compute_delta(k):
    """Computes delta from the block size k. Interpolation from the following
    data table:
    Source : https://bitbucket.org/malb/lwe-estimator/
    src/9302d4204b4f4f8ceec521231c4ca62027596337/estima
    tor.py?at=master&fileviewer=file-view-default
    :k: integer
    estimator.py table:
    """

    small = {0: 1e20, 1: 1e20, 2: 1.021900, 3: 1.020807, 4: 1.019713, 5: 1.018620,
             6: 1.018128, 7: 1.017636, 8: 1.017144, 9: 1.016652, 10: 1.016160,
             11: 1.015898, 12: 1.015636, 13: 1.015374, 14: 1.015112, 15: 1.014850,
             16: 1.014720, 17: 1.014590, 18: 1.014460, 19: 1.014330, 20: 1.014200,
             21: 1.014044, 22: 1.013888, 23: 1.013732, 24: 1.013576, 25: 1.013420,
             26: 1.013383, 27: 1.013347, 28: 1.013310, 29: 1.013253, 30: 1.013197,
             31: 1.013140, 32: 1.013084, 33: 1.013027, 34: 1.012970, 35: 1.012914,
             36: 1.012857, 37: 1.012801, 38: 1.012744, 39: 1.012687, 40: 1.012631,
             41: 1.012574, 42: 1.012518, 43: 1.012461, 44: 1.012404, 45: 1.012348,
             46: 1.012291, 47: 1.012235, 48: 1.012178, 49: 1.012121, 50: 1.012065}

    if k != round(k):
        x = k - floor(k)
        d1 = compute_delta(floor(k))
        d2 = compute_delta(floor(k) + 1)
        return x * d2 + (1 - x) * d1

    k = int(k)
    if k < 50:
        return small[k]
    else:
        delta = GH_sv_factor_squared(k)**(1. / (2. * k - 2.))
        return delta


def bkzgsa_gso_len(logvol, i, d, beta=None, delta=None):
    if delta is None:
        delta = compute_delta(beta)

    return delta**(d - 1 - 2 * i) * exp(logvol / d)

def qary_simulator(f, d, n, q, beta, xi=1, tau=1, dual=False, ignore_qary=False):
    """
    Reduced lattice shape calling ``f``.

    :param d: Lattice dimension.
    :param n: The number of `q` vectors is `d-n-1`.
    :param q: Modulus `q`
    :param beta: Block size β.
    :param xi: Scaling factor ξ for identity part.
    :param tau: Kannan factor τ.
    :param dual: perform reduction on the dual.
    :param ignore_qary: Ignore the special q-ary structure (forget q vectors)

    """

    assert 2 <= beta <= d

    if not tau:
        r = [q**2] * (d - n) + [xi**2] * n
    else:
        r = [q**2] * (d - n - 1) + [xi**2] * n + [tau**2]

    if ignore_qary:
        r = GSA(d, n, q, 2, xi=xi, tau=tau)

    if dual:
        # 1. reverse and reflect the basis (go to dual)
        r = [1 / r_ for r_ in reversed(r)]
        # 2. simulate reduction on the dual basis
        r = f(r, beta)
        # 3. reflect and reverse the basis (go back to primal)
        r = [1 / r_ for r_ in reversed(r)]
        return r
    else:
        return f(r, beta)





def CN11(d, n, q, beta, xi=1, tau=0, dual=False, ignore_qary=False):
    """
    Reduced lattice shape using simulator from [AC:CheNgu11]_

    :param d: Lattice dimension.
    :param n: The number of `q` vectors is `d-n-1`.
    :param q: Modulus `q`
    :param beta: Block size β.
    :param xi: Scaling factor ξ for identity part.
    :param tau: Kannan factor τ.
    :param dual: perform reduction on the dual.
    :param ignore_qary: Ignore the special q-ary structure (forget q vectors)
    :returns: squared Gram-Schmidt norms

    """

    from fpylll import BKZ
    from fpylll.tools.bkz_simulator import simulate

    assert 2 <= beta <= d

    def f(r, beta):
        return simulate(r, BKZ.EasyParam(beta))[0]

    return qary_simulator(f=f, d=d, n=n, q=q, beta=beta, xi=xi, tau=tau, dual=dual, ignore_qary=ignore_qary)

def bkzgsa_gso_len(logvol, i, d, beta=None, delta=None):
    if delta is None:
        delta = compute_delta(beta)

    return delta**(d - 1 - 2 * i) * exp(logvol / d)

def plot_gso(r, *args, **kwds):
    return line([(i, r_,) for i, r_ in enumerate(r)], *args, **kwds)

def discrete_gaussian_std(sigma, tailcut=8):
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    B = max(1, ceil(tailcut * sigma))
    xs = np.arange(-B, B + 1, dtype=np.float64)
    ws = np.exp(-(xs**2) / (2.0 * sigma * sigma))
    Z = ws.sum()
    var = (xs**2 @ ws) / Z
    return np.sqrt(var)

#Thm. 4.1 from https://eprint.iacr.org/2025/1910.pdf
def find_beta(d, n, q, st_dev_e):
    for beta in range(2, d//2, 1):
        gso_len = log(bkzgsa_gso_len(n*log(q), d-beta, d, beta=beta))
        lhs  = 0.5*log(beta)+log(st_dev_e)
        if lhs < 0.9*gso_len:
            return beta
    return float('inf')

# as per Lemma 4.2 from https://eprint.iacr.org/2019/1019.pdf
# assumes dist_param_e=dist_param_s="ternary"
#input: R-factor from G.r() storing ||b_i*||^2
def adm_probability(r,alpha=None,q=None,mode="ternary"):
    p = 1
    if mode=="ternary":
        r = [ exp(2*rr) for rr in r ]
        for i in range(len(r)):
            p*=(1-2./(3*(sqrt(r[i])+1)))
    if mode in ("discrete_gaussian", "gaussian"):
        sqrtpi = sqrt(pi)
        for i in range(len(r)):
            risqrtpi_aq = r[i] * sqrtpi / (alpha*q)
            try:
                p *= erf(risqrtpi_aq) + alpha*q/r[i] * ( ( exp( -(risqrtpi_aq)**2 ) - 1 ) / pi )
            except Exception as err:
                print(err)
                return 0
    return p

# #only for ternary
# def find_beta_for_adm(d, n, q, st_dev_e, target_succ_probability):
#     for beta in range(2, d-10, 1):
#         gso_len = ZGSA(d,n,q, beta)
#         if adm_probability([exp(RR(2*i)) for i in gso_len])>=target_succ_probability:
#             return beta
#     return float('inf')

# def find_beta_for_adm_proj(d, n, q, st_dev_e, target_succ_probability,cd):
#     for beta in range(2, d-10, 1):
#         gso_len = ZGSA(d,n,q, beta)
#         if adm_probability([exp(min(128,RR(2*i))) for i in gso_len[-cd:]])>=target_succ_probability:
#             return beta
#     return float('inf')

def find_beta_for_adm_proj(d, n, q, dist_e, st_dev_e, target_succ_probability, cd):
    """
    Find the smallest beta in [2, d-10) such that

        adm_probability([exp(min(128, 2*rr)) for rr in ZGSA(...)[-cd:]])
            >= target_succ_probability

    using binary search.

    Notes:
    - This assumes the predicate is monotone in beta.
    - `st_dev_e` is retained in the signature for compatibility, as in your original code.
    """

    lo = 2
    hi = d - 10  # exclusive upper bound in the original range(2, d-10)

    if lo >= hi:
        return inf

    def succeeds(beta):
        gso_len = CN11(d, n, q, beta)
        gso_len = [ sqrt(rr) for rr in gso_len ]
        alpha = st_dev_e / q * sqrt(2*pi)
        tmp = adm_probability([rr for rr in gso_len[-cd:]],alpha=alpha,q=q, mode=dist_e ) #gaussian
        # print(f"beta={beta}, p={tmp}")
        return tmp >= target_succ_probability


    # Binary search for the leftmost beta that succeeds
    while lo < hi:
        mid = (lo + hi) // 2
        if succeeds(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo

def f_to_integrate(d, r):
    j_to_integrate = lambda y, z: (1-y**2)**((d-3)/2.)
    if r<0.5:
        res =  np.array( integrate.dblquad(j_to_integrate, -r-1, r-1, -1, lambda z: z+r) ) +\
             + np.array( integrate.dblquad(j_to_integrate, r-1, -r, lambda z: z-r, lambda z: z+r) )
        assert(abs(res[1])<10e-4)
        return res[0]
    else:
        res  = integrate.dblquad(j_to_integrate, -r-1, -r, -1, lambda z: z+r)
        assert(abs(res[1])<10e-4)
        return res[0]


# as per Eq(8) from https://eprint.iacr.org/2016/089.pdf
def adm_probability2(d, r, bdd_er_norm):
    p = 1
    beta_fn_value = beta((d-1)/2., 1./2)
    for i in range(len(r)):
        r_sq = sqrt(r[i])
        r_scaled = r_sq/(2*bdd_er_norm)
        p*=(1-f_to_integrate(d, r_scaled)/(r_scaled*beta_fn_value))
    return p