import os
import time
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from fpylll import IntegerMatrix, GSO, FPLLL

from lwe_gen import generateLWEInstances
from lattice_reduction import LatticeReduction
from zgsa_fast import find_beta_for_adm_proj


# ----------------------------
# Configuration
# ----------------------------
FPLLL.set_precision(208)

n, m, q = 100, 100, 18839
dist_s, dist_param_s = "discrete_gaussian", 1.0
dist_e, dist_param_e = "discrete_gaussian", 1.0

kappa = 20
# Number of independent lattices / experiments
n_lattices = 10
n_targets = 1000
target_succ_probability = 0.005 #controls the blocksize of BKZ

a, b, n_dims = 40, min(100, n + m - kappa), 4
cds = np.asarray(np.round(np.linspace(a, b, n_dims)), dtype=int)

bkz_tours = 5
lll_size = 64

# Parallelism over lattices
max_workers = min(n_lattices, os.cpu_count() or 1)

# Output directory
experiments_dir = Path("experiments")
experiments_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def project_onto_last(G, v, cd):
    assert cd <= G.d, f"Too large dim {cd}>{G.d}"
    v_gh = np.asarray(G.from_canonical(v), dtype=float)
    v_gh[:-cd] = 0
    return np.asarray(G.to_canonical(v_gh), dtype=float)


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


def compute_beta(n, m, q, kappa, dist_param_e,cd):
    """
    Keep the original beta logic.
    """
    # beta = find_beta(n + m - kappa, n, q, 3 * dist_param_e) #use this for ternary
    beta = find_beta_for_adm_proj(
        n+m-kappa, n, q, dist_e, dist_param_e, 
        target_succ_probability=target_succ_probability, 
        cd=cd)  #use this for gauss
    if beta > n:
        beta = 50
    return int(beta)


def reduce_lattice(H, beta, lll_size, bkz_tours):
    """
    Apply the same preprocessing / reduction strategy as your script.
    """
    LatRed_instance = LatticeReduction(H)

    if beta > 49:
        _ = LatRed_instance(
            lll_size=lll_size,
            delta=0.99,
            cores=1,
            beta=42,
            bkz_tours=2,
        )

    Hred = LatRed_instance(
        lll_size=lll_size,
        delta=0.99,
        cores=1,   # safer for multiprocessing; avoids oversubscription
        beta=beta,
        bkz_tours=bkz_tours,
    )
    return Hred


def as_python_int(x):
    return int(x) if isinstance(x, (np.integer,)) else x


def run_one_lattice(exp_id):
    """
    Run the full experiment for one independently generated lattice
    and its n_targets corresponding LWE instances.
    Returns a serializable dictionary and also dumps it to experiments/.
    """
    t0 = time.time()

    # 1) Generate one LWE matrix A and n_targets corresponding instances
    A, _, bse = generateLWEInstances(
        n, m, q,
        dist_s, dist_param_s,
        dist_e, dist_param_e,
        n_targets,
    )
    assert len(bse) == n_targets

    # 2) Build lattice basis
    B = build_lwe_basis(A, n, m, q)

    # 3) Split basis as in original code
    Htmp = B[:len(B) - kappa]
    H = IntegerMatrix.from_matrix([row[:len(B) - kappa] for row in Htmp])
    C = np.array([row[:len(B) - kappa] for row in B[len(B) - kappa:]], dtype=np.int64)

    # 4) Compute beta
    beta = compute_beta(n, m, q, kappa, dist_param_e, cds[0])

    # 5) Reduce basis
    Hred = reduce_lattice(H, beta, lll_size, bkz_tours)

    # 6) Build GSO
    G = GSO.Mat(IntegerMatrix.from_matrix(Hred), float_type="mpfr")
    G.update_gso()

    # 7) Babai-lift filtering
    bse_survivors = list(bse)
    babai_lift_success = 0

    for i in range(len(bse_survivors) - 1, -1, -1):
        b_vec, s_vec, e_vec = bse_survivors[i]

        sguess = s_vec[-kappa:]
        sguess_times_C = sguess @ C
        target = np.concatenate([b_vec, np.zeros(n - kappa, dtype=int)]) - sguess_times_C

        babai_res = G.babai(target)
        tshift = target - G.B.multiply_left(babai_res)

        diff = tshift - np.concatenate([e_vec, -s_vec[:-kappa]])
        if np.all(np.isclose(diff, 0.0, atol=1e-7)):
            babai_lift_success += 1
        else:
            del bse_survivors[i]

    # 8) Full-dimension admissibility
    full_dim_succ = 0
    for b_vec, s_vec, e_vec in bse_survivors:
        sguess = s_vec[-kappa:]

        w1 = np.concatenate([sguess[:kappa // 2], np.zeros(kappa // 2, dtype=int)])
        w2 = np.concatenate([np.zeros(kappa // 2, dtype=int), sguess[kappa // 2:]])

        target_w1 = np.concatenate([b_vec, np.zeros(n - kappa, dtype=int)]) - w1 @ C
        target_w2 = -w2 @ C

        babai_res_w1 = G.babai(target_w1)
        err_w1 = target_w1 - G.B.multiply_left(babai_res_w1)

        babai_res_w2 = G.babai(target_w2)
        err_w2 = target_w2 - G.B.multiply_left(babai_res_w2)

        err_w1_gs = np.asarray(G.from_canonical(err_w1), dtype=float)
        err_w2_gs = np.asarray(G.from_canonical(err_w2), dtype=float)

        lhs = np.asarray(G.from_canonical(np.concatenate([e_vec, -s_vec[:-kappa]])), dtype=float) - err_w1_gs
        rhs = err_w2_gs

        diff = lhs - rhs
        if np.all(np.isclose(diff, 0.0, atol=1e-7)):
            full_dim_succ += 1

    # 9) Projected admissibility by cd
    results = {}
    for cd in cds:
        cd = int(cd)
        cd_dim_succ = 0

        for b_vec, s_vec, e_vec in bse_survivors:
            sguess = s_vec[-kappa:]

            w1 = np.concatenate([sguess[:kappa // 2], np.zeros(kappa // 2, dtype=int)])
            w2 = np.concatenate([np.zeros(kappa // 2, dtype=int), sguess[kappa // 2:]])

            target_w1 = np.concatenate([b_vec, np.zeros(n - kappa, dtype=int)]) - w1 @ C
            target_w2 = -w2 @ C

            target_w1_proj = project_onto_last(G, target_w1, cd)
            target_w2_proj = project_onto_last(G, target_w2, cd)

            babai_res_w1_proj = G.babai(target_w1_proj)
            err_w1_proj = target_w1_proj - G.B.multiply_left(babai_res_w1_proj)

            babai_res_w2_proj = G.babai(target_w2_proj)
            err_w2_proj = target_w2_proj - G.B.multiply_left(babai_res_w2_proj)

            err_w1_proj_gs = np.asarray(G.from_canonical(err_w1_proj), dtype=float)[-cd:]
            err_w2_proj_gs = np.asarray(G.from_canonical(err_w2_proj), dtype=float)[-cd:]

            lhs_proj = np.asarray(
                G.from_canonical(np.concatenate([e_vec, -s_vec[:-kappa]])),
                dtype=float
            )[-cd:] - err_w1_proj_gs
            rhs_proj = err_w2_proj_gs

            diff_proj = lhs_proj - rhs_proj
            if np.all(np.isclose(diff_proj, 0.0, atol=1e-7)):
                cd_dim_succ += 1

        results[cd] = {
            "bab_fpylll_proj": int(cd_dim_succ),
        }

    elapsed_s = time.time() - t0

    # 10) Collect everything that used to be printed, plus beta
    experiment_dict = {
        "exp_id": int(exp_id),
        "params": {
            "n": int(n),
            "m": int(m),
            "q": int(q),
            "kappa": int(kappa),
            "n_targets": int(n_targets),
            "dist_s": dist_s,
            "dist_param_s": float(dist_param_s),
            "dist_e": dist_e,
            "dist_param_e": float(dist_param_e),
            "cds": [int(x) for x in cds],
            "bkz_tours": int(bkz_tours),
            "lll_size": int(lll_size),
        },
        "beta": int(beta),
        "babai_lift_success": int(babai_lift_success),
        "full_dim_succ": int(full_dim_succ),
        "results": {
            int(cd): {
                k: as_python_int(v) for k, v in val.items()
            }
            for cd, val in results.items()
        },
        "elapsed_s": float(elapsed_s),
    }

    # 11) Dump per-experiment pickle
    out_path = experiments_dir / f"exp_{exp_id:04d}_{n}_{q}_{kappa}_{dist_s}_{dist_param_s}_{dist_e}_{dist_param_e}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(experiment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return experiment_dict


def main():
    print(f"Running {n_lattices} independent lattices with {max_workers} workers.")
    print(f"Results will be written to: {experiments_dir.resolve()}")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_lattice, exp_id) for exp_id in range(n_lattices)]

        for fut in as_completed(futures):
            res = fut.result()
            all_results.append(res)

            print(
                f"[exp {res['exp_id']}] beta={res['beta']}, "
                f"babai_lift_success={res['babai_lift_success']}, "
                f"full_dim_succ={res['full_dim_succ']}"
            )
            print(res["results"], flush=True)

    # Optional combined dump
    combined = {
        "n_lattices": int(n_lattices),
        "all_results": all_results,
    }
    combined_path = experiments_dir / "all_experiments.pkl"
    with open(combined_path, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved combined results to {combined_path.resolve()}")


if __name__ == "__main__":
    main()