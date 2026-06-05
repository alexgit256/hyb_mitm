from time import perf_counter

from fpylll import *
from fpylll.util import gaussian_heuristic
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ as BKZ_FPYLLL
from fpylll.algorithms.bkz2 import BKZReduction

import numpy as np
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gammaln, betainc
from scipy.integrate import quad

def uniform_in_ball(num_points, dimension, radius=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    random_directions = rng.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)

    random_radii = rng.random(num_points) ** (1.0 / dimension)

    return radius * (random_directions * random_radii).T

def uniform_in_fund_par(num_points, G, rng=None):
    """
    Sample uniformly from the fundamental parallelepiped.

    Returns canonical coordinates, not Gram-Schmidt coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()

    dimension = G.B.nrows

    gs_coordinates = rng.uniform(
        -0.5, 0.5,
        size=(num_points, dimension)
    )

    canon_coordinates = np.array([
        np.array(G.to_canonical(tuple(float(vv) for vv in v)), dtype=float)
        for v in gs_coordinates
    ])

    return canon_coordinates

def modulo_fund_parall(points, G):
    out = []

    for v in points:
        v = np.asarray(v, dtype=float)

        c = G.babai(tuple(float(vv) for vv in v))
        vbab = np.array(G.B.multiply_left(c), dtype=float)

        out.append(v - vbab)

    return np.asarray(out)

def modulo_voronoi_cell(points, G, gh=None, radius_factor=1.14):
    """
    Reduce canonical vectors modulo the Voronoi cell.

    Input:
        points: array of canonical-coordinate vectors.
        G:      fpylll GSO.Mat.
        gh:     squared Gaussian heuristic scale.

    Output:
        reduced points in canonical coordinates.
    """
    if gh is None:
        gh = gaussian_heuristic(G.r())

    enum = Enumeration(G, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS)

    out = []

    for v in points:
        v = np.asarray(v, dtype=float)

        target = G.from_canonical(tuple(float(vv) for vv in v))

        R2 = radius_factor * gh

        # In rare cases the bound may be too small.
        # If no vector is found, enlarge the radius.
        while True:
            sols = enum.enumerate(
                0, G.d,
                R2,
                0,
                target=target
            )

            if len(sols) > 0:
                break

            R2 *= 2.0

        ccvp = sols[0][1]

        vcvp = np.array(G.B.multiply_left(ccvp), dtype=float)
        out.append(v - vcvp)

    return np.asarray(out)

def build_lattice_gso(n, qary_bits=11, bkz_block_sizes=[25], bkz_max_loops=4,
                      float_type="d", seed=0):
    """
    Build one random q-ary lattice, LLL-reduce it, optionally BKZ-reduce it,
    and return its GSO object.
    """
    FPLLL.set_random_seed(int(seed))

    B = IntegerMatrix(n, n)
    B.randomize("qary", k=n//2, bits=qary_bits)

    G = GSO.Mat(B, float_type=float_type)
    G.update_gso()

    lll_obj = LLL.Reduction(G)
    lll_obj()
    G.update_gso()

    if n > 25 and bkz_block_sizes is not None:
        bkz = BKZReduction(G)
        for beta in bkz_block_sizes:
            par = BKZ_FPYLLL.Param(
                beta,
                strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
                max_loops=bkz_max_loops,
                flags=BKZ_FPYLLL.MAX_LOOPS
            )
    
            bkz(par)

        G = bkz.M
        G.update_gso()

    return G

def experiment_one_lattice(args):
    """
    Worker function.

    Runs the whole gamma sweep for one independently sampled lattice.
    """
    (
        lattice_id,
        seed,
        n,
        qary_bits,
        beta,
        bkz_max_loops,
        num_points,
        gammas,
        float_type,
        reuse_noise
    ) = args
    t0 = perf_counter()

    rng = np.random.default_rng(int(int(seed) + 10**6))

    G = build_lattice_gso(
        n=n,
        qary_bits=qary_bits,
        bkz_block_sizes=beta,
        bkz_max_loops=bkz_max_loops,
        float_type=float_type,
        seed=seed
    )

    gh2 = gaussian_heuristic(G.r())
    gh_radius = gh2**0.5

    # Sample v uniformly in the fundamental parallelepiped,
    # then map it measure-preservingly to the Voronoi cell.
    V = uniform_in_fund_par(num_points, G, rng=rng)

    # 
    VmodVor = modulo_voronoi_cell(V, G, gh=gh2)

    # Optional variance reduction:
    # use the same unit-ball samples for every gamma on this lattice.
    if reuse_noise:
        W_unit = uniform_in_ball(
            num_points,
            dimension=n,
            radius=1.0,
            rng=rng
        )
    else:
        W_unit = None

    results = []

    for gamma in gammas:
        gamma = float(gamma)

        if reuse_noise:
            W = gamma * gh_radius * W_unit
        else:
            W = uniform_in_ball(
                num_points,
                dimension=n,
                radius=gamma * gh_radius,
                rng=rng
            )

        T = VmodVor + W
        TmodVoronoi = modulo_voronoi_cell(T, G, gh=gh2)

        diff = T - TmodVoronoi

        eps = 1e-8
        changed = np.linalg.norm(diff, axis=1) > eps

        cvp_successes = int(np.count_nonzero(~changed))
        cvp_trials = int(len(changed))

        success_probability = cvp_successes / cvp_trials

        # vs just babai
        #NOTE: vectors from V are already uniform mod fund. parall.

        TmeetBab = V + W
        TmodBab = modulo_fund_parall(TmeetBab, G)

        diff_bab = TmeetBab - TmodBab

        eps = 1e-8
        changed_bab = np.linalg.norm(diff_bab, axis=1) > eps

        bab_successes = int(np.count_nonzero(~changed_bab))
        bab_trials = int(len(changed_bab))

        success_probability_bab = bab_successes / bab_trials

        results.append({
            "lattice_id": lattice_id,
            "seed": seed,
            "gamma": gamma,

            "success_probability": float(success_probability),
            "success_probability_bab": float(success_probability_bab),

            "cvp_successes": cvp_successes,
            "cvp_trials": cvp_trials,

            "bab_successes": bab_successes,
            "bab_trials": bab_trials,

            "num_points": num_points,
            "n": n,
            "beta": beta,
        })
    
    print(f"{lattice_id} finished in {()-t0}")
    return results

# - - - BALL
def log_unit_ball_volume(n):
    """
    log Vol(B_1^n).
    """
    return (n / 2.0) * np.log(np.pi) - gammaln(n / 2.0 + 1.0)


def ball_volume(n, R):
    return np.exp(log_unit_ball_volume(n) + n * np.log(R))


def equal_volume_radius(n, det=1.0):
    """
    Radius R such that Vol(B_R^n) = det.
    """
    log_V1 = log_unit_ball_volume(n)
    return np.exp((np.log(det) - log_V1) / n)

def cap_volume(n, a, h):
    """
    Volume of an n-dimensional spherical cap of height h
    cut from a ball of radius a.

    h = 0 gives 0.
    h = a gives half the ball.
    h = 2a gives the whole ball.
    """
    if h <= 0.0:
        return 0.0

    full = ball_volume(n, a)

    if h >= 2.0 * a:
        return full

    # For numerical stability, use symmetry for caps larger than half.
    if h > a:
        return full - cap_volume(n, a, 2.0 * a - h)

    z = 1.0 - (1.0 - h / a)**2

    return 0.5 * full * betainc((n + 1.0) / 2.0, 0.5, z)

def ball_intersection_volume(n, R, r, d):
    """
    Volume of intersection of two n-balls:
        B_R(0) and B_r(center at distance d).
    """
    VR = ball_volume(n, R)
    Vr = ball_volume(n, r)

    if d <= 0.0:
        return min(VR, Vr)

    # B_r fully inside B_R
    if d + r <= R:
        return Vr

    # B_R fully inside B_r
    if d + R <= r:
        return VR

    # Disjoint, not relevant for d <= R
    if d >= R + r:
        return 0.0

    x = (d*d + R*R - r*r) / (2.0 * d)

    h_R = R - x
    h_r = r - (d - x)

    return cap_volume(n, R, h_R) + cap_volume(n, r, h_r)

def ball_model_success_probability(n, gamma, det=1.0):
    """
    Average success probability for:
        v uniform in B_R
        w uniform in B_r
        success iff v + w in B_R

    with r = gamma * R.
    P_success =
    ∫₀ᴿ [ Vol(B_R ∩ (d e₁ + B_r)) / Vol(B_r) ] · n d^{n-1}/R^n dd.
    """
    R = equal_volume_radius(n, det=det)
    r = gamma * R

    Vr = ball_volume(n, r)

    def integrand(d):
        inter = ball_intersection_volume(n, R, r, d)
        radial_density = n * d**(n - 1) / R**n
        return (inter / Vr) * radial_density

    val, err = quad(
        integrand,
        0.0,
        R,
        epsabs=1e-10,
        epsrel=1e-8,
        limit=200
    )

    return val
# - - - END BALL

def aggregate_results(df, variation_mode="lattices", error_as="sem"):
    """
    variation_mode:
        "lattices" : variation of per-lattice success probabilities.
        "targets"  : pooled Bernoulli variation over all targets.

    error_as:
        "sem" : standard error of the mean.
        "std" : standard deviation.
    """

    if variation_mode == "lattices":
        agg = (
            df.groupby("gamma")
              .agg(
                  cvp_mean=("success_probability", "mean"),
                  cvp_std=("success_probability", "std"),
                  cvp_count=("success_probability", "count"),

                  bab_mean=("success_probability_bab", "mean"),
                  bab_std=("success_probability_bab", "std"),
                  bab_count=("success_probability_bab", "count"),
              )
              .reset_index()
        )

        if error_as == "sem":
            agg["cvp_yerr"] = agg["cvp_std"] / np.sqrt(agg["cvp_count"])
            agg["bab_yerr"] = agg["bab_std"] / np.sqrt(agg["bab_count"])
        elif error_as == "std":
            agg["cvp_yerr"] = agg["cvp_std"]
            agg["bab_yerr"] = agg["bab_std"]
        else:
            raise ValueError("error_as must be 'sem' or 'std'")

        return agg

    elif variation_mode == "targets":
        agg = (
            df.groupby("gamma")
              .agg(
                  cvp_successes=("cvp_successes", "sum"),
                  cvp_trials=("cvp_trials", "sum"),

                  bab_successes=("bab_successes", "sum"),
                  bab_trials=("bab_trials", "sum"),
              )
              .reset_index()
        )

        agg["cvp_mean"] = agg["cvp_successes"] / agg["cvp_trials"]
        agg["bab_mean"] = agg["bab_successes"] / agg["bab_trials"]

        # Bernoulli standard deviation over individual targets.
        agg["cvp_std"] = np.sqrt(agg["cvp_mean"] * (1.0 - agg["cvp_mean"]))
        agg["bab_std"] = np.sqrt(agg["bab_mean"] * (1.0 - agg["bab_mean"]))

        if error_as == "sem":
            agg["cvp_yerr"] = agg["cvp_std"] / np.sqrt(agg["cvp_trials"])
            agg["bab_yerr"] = agg["bab_std"] / np.sqrt(agg["bab_trials"])
        elif error_as == "std":
            agg["cvp_yerr"] = agg["cvp_std"]
            agg["bab_yerr"] = agg["bab_std"]
        else:
            raise ValueError("error_as must be 'sem' or 'std'")

        return agg

    else:
        raise ValueError("variation_mode must be 'lattices' or 'targets'")

if __name__ == "__main__":
    # Experiment parameters

    n = 45
    num_points = 4096

    qary_bits = 17
    betas = [25, n]
    bkz_max_loops = 4
    float_type = "d"

    num_lattices = 80
    num_workers = 8

    gammas = np.linspace(0.05, 0.6, 12)

    # If True, each lattice uses the same noise directions/radii across gammas,
    # only scaled by gamma. This makes the curve less noisy.
    reuse_noise = True

    base_seed = 1337

    jobs = []

    for lattice_id in range(num_lattices):
        seed = base_seed + lattice_id

        jobs.append((
            lattice_id,
            seed,
            n,
            qary_bits,
            betas,
            bkz_max_loops,
            num_points,
            gammas,
            float_type,
            reuse_noise
        ))

    # In Sage/Jupyter on Linux, "fork" is usually the most convenient.
    # If this fails in your environment, try ctx = mp.get_context("spawn"),
    # but then functions may need to live in an importable .py file.

    ctx = mp.get_context("fork")

    with ctx.Pool(processes=num_workers) as pool:
        nested_results = pool.map(experiment_one_lattice, jobs)

    rows = [row for lattice_result in nested_results for row in lattice_result]

    df = pd.DataFrame(rows)
    df

    variation_mode = "targets"
    error_as = "sem"

    agg = aggregate_results(
        df,
        variation_mode=variation_mode,
        error_as=error_as
    )

    print( agg )

    betas_tag = "_".join(str(b) for b in betas)

    gammas_dense = np.linspace(float(min(gammas)), float(max(gammas)), 100)

    ball_curve = np.array([
        ball_model_success_probability(n, gamma, det=1.0)
        for gamma in gammas_dense
    ])

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        agg["gamma"],
        agg["cvp_mean"],
        yerr=agg["cvp_yerr"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        label="CVP / Voronoi"
    )

    plt.errorbar(
        agg["gamma"],
        agg["bab_mean"],
        yerr=agg["bab_yerr"],
        marker="s",
        capsize=4,
        linewidth=1.5,
        label="Babai / fundamental parallelepiped"
    )

    plt.plot(
        gammas_dense,
        ball_curve,
        linestyle="--",
        linewidth=2.0,
        label="equal-volume ball model"
    )

    plt.xlabel(r"$\gamma$")
    plt.ylabel("successful decoding probability")
    plt.title(
        f"Decoding success over {num_lattices} lattices, "
        f"n={n}, betas={betas},\n"
        f" points/lattice={num_points}, variation={variation_mode}/{error_as}"
    )

    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"pltvdc{n}_{betas_tag}_with_ball.png", dpi=384, bbox_inches="tight")
    plt.show()
    