import numpy as np
from math import lgamma
from lattice_reduction import LatticeReduction
from zgsa_fast import  find_beta
import statistics
from fpylll import IntegerMatrix, GSO
from time import perf_counter

# ----------------------------
# Helpers
# ----------------------------

def as_python_int(x):
    return int(x) if isinstance(x, (np.integer,)) else x

def project_onto_last(G, v, cd):
    assert cd <= G.d, f"Too large dim {cd}>{G.d}"
    v_gh = np.asarray(G.from_canonical(v), dtype=float)
    v_gh[:-cd] = 0
    return np.asarray(G.to_canonical(v_gh), dtype=float)

def gs_projected_canonical_norm(G, v, cd):
    """
    Take a canonical vector v, project it onto the last cd Gram-Schmidt coordinates,
    map back to canonical coordinates, and return its Euclidean norm.
    """
    vgs = np.asarray(G.from_canonical(v), dtype=float).copy()
    vgs[:-cd] = 0.0
    vproj = np.asarray(G.to_canonical(vgs), dtype=float)
    return float(np.sqrt(vproj @ vproj))

def split_secret_guess(sguess, kappa):
    half = kappa // 2
    w1 = np.concatenate([sguess[:half], np.zeros(len(sguess) - half, dtype=int)])
    w2 = np.concatenate([np.zeros(half, dtype=int), sguess[half:]])
    return w1, w2

def lwe_target(b_vec, guess, C, n, kappa):
    return np.concatenate([b_vec, np.zeros(n - kappa, dtype=int)]) - guess @ C

def bdd_error_vector(e_vec, s_vec, kappa):
    return np.concatenate([e_vec, -s_vec[:-kappa]])

def babai_residual(G, target):
    babai_res = G.babai(target)
    return target - G.B.multiply_left(babai_res)

def make_gso(Hred, float_type="mpfr"):
    G = GSO.Mat(IntegerMatrix.from_matrix(Hred), float_type=float_type)
    G.update_gso()
    return G

def is_babai_lift_success(G, b_vec, s_vec, e_vec, C, n, kappa, atol=1e-7):
    sguess = s_vec[-kappa:]
    target = lwe_target(b_vec, sguess, C, n, kappa)

    residual = babai_residual(G, target)
    expected = np.concatenate([e_vec, -s_vec[:-kappa]])

    return np.all(np.isclose(residual - expected, 0.0, atol=atol))

def filter_babai_lift_survivors(G, bse, C, n, kappa, atol=1e-7):
    survivors = []
    observed_full_norms = []

    for b_vec, s_vec, e_vec in bse:
        if is_babai_lift_success(G, b_vec, s_vec, e_vec, C, n, kappa, atol=atol):
            survivors.append((b_vec, s_vec, e_vec))
            esvec = np.concatenate([e_vec, s_vec[-kappa:]])
            observed_full_norms.append(float(np.sqrt(esvec @ esvec)))

    return survivors, len(survivors), observed_full_norms

def summarize_prediction(pred, obs):
    """
    pred : scalar prediction
    obs  : list of observed values
    """
    if not obs:
        return {
            "count": 0,
            "pred": float(pred),
            "mean_obs": None,
            "std_obs": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "rel_mae_to_mean": None,
        }

    obs = [float(x) for x in obs]
    errs = [x - pred for x in obs]
    mae = sum(abs(e) for e in errs) / len(errs)
    rmse = np.sqrt(sum(e * e for e in errs) / len(errs))
    bias = sum(errs) / len(errs)
    mean_obs = sum(obs) / len(obs)
    std_obs = statistics.pstdev(obs) if len(obs) > 1 else 0.0

    return {
        "count": len(obs),
        "pred": float(pred),
        "mean_obs": float(mean_obs),
        "std_obs": float(std_obs),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "rel_mae_to_mean": float(mae / mean_obs) if abs(mean_obs) > 1e-15 else None,
    }

def build_lwe_basis(A, n, m, q):
    """
    Build the standard q-ary lattice basis:
        [ q I_m ]
        [  A    I_n ]
    in row form.
    """
    B = [[0 for _ in range(m + n)] for _ in range(m + n)]

    for i in range(m):
        B[i][i] = int(q)

    for i in range(m, m + n):
        B[i][i] = 1

    for i in range(m, m + n):
        for j in range(m):
            B[i][j] = int(A[i - m, j])

    return B

def build_partitioned_basis(A, n, m, q, kappa):
    B = build_lwe_basis(A, n, m, q)

    # 3) Split basis as in original code
    Htmp = B[:len(B) - kappa]
    H = IntegerMatrix.from_matrix([row[:len(B) - kappa] for row in Htmp])
    C = np.array([row[:len(B) - kappa] for row in B[len(B) - kappa:]], dtype=np.int64)
    return H, C

def fill_zero_projected_results(
    result,
    remaining_cds,
    r_vec,
    z_shape,
    lens_proj_beta,
):
    #used in compute_projected_admissibility_by_cd
    for cd in remaining_cds:
        bdd_err_norm_proj = lens_proj_beta[cd]["pred"]

        result[cd] = {
            "successes": 0,
            "prob_exact_r": adm_probability2(cd, r_vec[-cd:], bdd_err_norm_proj),
            "prob_gsa": adm_probability2(cd, z_shape[-cd:], bdd_err_norm_proj),
            "prob_mitm_babai": mitm_babai_probability(
                r_vec[-cd:], bdd_err_norm_proj / sqrt(cd)
            ),
            "obs_norms": [],
        }

def compute_beta(n, m, q, kappa, dist_e, dist_param_e, cd):
    """
    Keep the original beta logic.
    """
    if dist_e=="ternary":
        beta = find_beta(n + m - kappa, n, q, 3 * dist_param_e) #use this for ternary
    if dist_e=="ternary_sparse":
        beta = find_beta(n + m - kappa, n, q, 3 * dist_param_e) #use this for ternary
    elif dist_e=="binomial":
        beta = find_beta(n + m - kappa, n, q, dist_param_e/2)
    elif dist_e=="binary":
        beta = find_beta(n + m - kappa, n, q, dist_param_e)
    elif dist_e in ["gaussian", "discrete_gaussian"]:
        beta = find_beta(n + m - kappa, n, q, discrete_gaussian_std(dist_param_e))
    else:
        raise NotImplementedError(f"Dist {dist_e} not supported")
    if beta > n:
        beta = 80
    return int(beta)

def reduce_lattice(H, beta, lll_size, bkz_tours, cores=1):
    """
    Apply the same preprocessing / reduction strategy as your script.
    """
    LatRed_instance = LatticeReduction(H)

    _ = LatRed_instance(
        lll_size=lll_size,
        delta=0.99,
        cores=1,
        beta=min(beta,49),
        bkz_tours=2,
    )

    if beta > 50:
        for bbeta in range(50,beta):
             print( f"Doing BKZ-{bbeta}" )
             t0 = perf_counter()
             _ = LatRed_instance(
                lll_size=lll_size,
                delta=0.99,
                cores=cores,
                beta=bbeta,
                bkz_tours=2,
            )
             print( f"BKZ-{bbeta} done in {perf_counter()-t0}" )

    t0 = perf_counter()
    
    Hred = LatRed_instance(
        lll_size=lll_size,
        delta=0.99,
        cores=1 if (beta < 55) else cores,
        beta=beta,
        bkz_tours=bkz_tours,
    )
    print( f"BKZ-{beta} done in {perf_counter()-t0}" )
    return Hred

def discrete_gaussian_std(sigma, tailcut=10):
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    B = max(1, np.ceil(tailcut * sigma))
    xs = np.arange(-B, B + 1, dtype=np.float64)
    ws = np.exp(-(xs**2) / (2.0 * sigma * sigma))
    ws /= ws.sum()
    var = np.dot(xs**2, ws)
    return np.sqrt(var)

def expected_bdd_err_norm(d, dist_e, dist_s, dist_param_s, dist_param_e, mode="mean"):
    assert dist_e == dist_s and dist_param_s == dist_param_e

    if dist_e == "discrete_gaussian":
        sigma1 = discrete_gaussian_std(dist_param_e) #* sqrt(2)
    elif dist_e == "binomial":
        sigma1 = np.sqrt(dist_param_e) / np.sqrt( 2.0 )
    elif dist_e == "ternary":
        sigma1 = np.sqrt(dist_param_e) * np.sqrt(2.)  # depends on parametrization
    elif dist_e == "binary":
        sigma1 = np.sqrt( dist_param_e )
    else:
        raise NotImplementedError(f"Distribution {dist_e!r} is not implemented.")

    if mode == "rms":
        return sigma1 * np.sqrt(d)
    elif mode == "mean":
        return sigma1 * np.sqrt(2.0) * np.exp(lgamma((d + 1) / 2.0) - lgamma(d / 2.0))
    elif mode == "mean_asymptotic":
        return sigma1 * np.sqrt(d - 0.5)
    else:
        raise ValueError("mode must be 'rms', 'mean', or 'mean_asymptotic'")