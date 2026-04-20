import os
import time
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from math import sqrt, log
import statistics

import numpy as np
from fpylll import IntegerMatrix, GSO, FPLLL

from lwe_gen import generateLWEInstances
from lattice_reduction import LatticeReduction
from zgsa_fast import find_beta_for_adm_proj, find_beta, bkzgsa_gso_len, adm_probability2, CN11
from math import sqrt, log

from PT25 import expected_proj_norm

# ----------------------------
# Helpers
# ----------------------------
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
    rmse = sqrt(sum(e * e for e in errs) / len(errs))
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


# def compute_beta(n, m, q, kappa, dist_e, dist_param_e, cd):
#     """
#     Keep the original beta logic.
#     """
#     beta = find_beta(n + m - kappa, n, q, 3 * dist_param_e) #use this for ternary
#     # beta = find_beta_for_adm_proj(
#     #     n+m-kappa, n, q, dist_e, dist_param_e, 
#     #     target_succ_probability=target_succ_probability, 
#     #     cd=cd)  #use this for gauss
#     if beta > n:
#         beta = 50
#     return int(beta)

def compute_beta(n, m, q, kappa, dist_e, dist_param_e, cd):
    """
    Keep the original beta logic.
    """
    if dist_e=="ternary":
        beta = find_beta(n + m - kappa, n, q, 3 * dist_param_e) #use this for ternary
    elif dist_e=="binomial":
        beta = find_beta(n + m - kappa, n, q, dist_param_e/2)
    elif dist_e in ["gaussian", "discrete_gaussian"]:
        beta = find_beta(n + m - kappa, n, q, discrete_gaussian_std(dist_param_e))
    else:
        raise NotImplementedError(f"Dist {dist_e} not supported")
    # beta+=25
    # beta = find_beta_for_adm_proj(
    #     n+m-kappa, n, q, dist_e, dist_param_e, 
    #     target_succ_probability=target_succ_probability, 
    #     cd=cd)  #use this for gauss
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

def expected_bdd_err_norm(d, dist_e, dist_s,  dist_param_s, dist_param_e):
    """
    d : dimension of the bdd error (projected or non-projected)
    return: expected Euclidean norm of the bdd error vector
    TODO: generalize to different distributions
    """
    assert dist_e==dist_s and dist_param_s==dist_param_e 
    if dist_e=="ternary":
        return sqrt(d*2*dist_param_s)
    elif dist_e=="discrete_gaussian":
        return dist_param_e*sqrt(d)
    elif dist_e=="binomial":
        return sqrt(d)*dist_param_e/2. #ToDo: check: centered binomial variance: eta/2

    else: raise NotImplementedError(f"Distribution {dist_param_s!r} is not implemented in expected_bdd_err_norm.")

# ----------------------------
# Configuration
# ----------------------------
FPLLL.set_precision(208)

n, m, q = 100, 100, 3299
dist_s, dist_param_s = "binomial", 2
dist_e, dist_param_e = "binomial", 2

kappa = 25
# Number of independent lattices / experiments
n_lattices = 8
n_targets = 100
target_succ_probability = 0.005 #controls the blocksize of BKZ

a, b, n_dims = 40, min(100, n + m - kappa), 4
# cds = np.asarray(np.round(np.linspace(a, b, n_dims)), dtype=int)
cds = [50,75]
print("cd values:", cds)

bkz_tours = 5
lll_size = 64
# Compute beta
beta_s = compute_beta(n, m, q, kappa, dist_e, dist_param_e, cds[0]) + 6
beta_values = [beta_s+i*10 for i in range(5) if beta_s+i*10<65]
print("beta values:", beta_values)

# Parallelism over lattices
max_workers = 8 #min(n_lattices, os.cpu_count() or 1)

# Output directory
experiments_dir = Path("experiments")
experiments_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Main function
# ----------------------------


def run_one_lattice(exp_id, beta_values):
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

    # dictionary to collect statistic on full lattice
    # [# babai success on full dim, #succ admissibility on full dim, estimated adm. succ using exact R, estimated adm. succ using GSA]
    stats_full = dict(  [ (beta, [0, 0, 0, 0] ) for beta in beta_values]  )

    # dictionary to collect statistic on projected lattices
    #for each beta and each cd, collect the same data as for stats_full except # babai success on full dim
    stats_proj = dict( [ (beta, dict([ (int(cd), [0, 0, 0]) for cd in cds ])) for beta in beta_values] )

    lens_full = {
        beta: {
            "pred": expected_bdd_err_norm(n + m - kappa, dist_e, dist_s, dist_param_s, dist_param_e), #2025/2195 is irrelevant w/o projection
            "obs": [],
        }
        for beta in beta_values
    }


    # lens_proj = {
    #     beta: {
    #         int(cd): {
    #             "pred":expected_proj_norm(n+m-kappa,lens_full[beta]["pred"],cd),   # "pred": expected_bdd_err_norm(int(cd), dist_e, dist_s, dist_param_s, dist_param_e),
    #             "obs": [],
    #         }
    #         for cd in cds
    #     }
    #     for beta in beta_values
    # }

    lens_proj = {
        beta: {
            int(cd): {
                "pred":expected_proj_norm(n+m-kappa,lens_full[beta]["pred"],cd), 
                "obs": [],
            }
            for cd in cds
        }
        for beta in beta_values
    }

    for beta in beta_values:

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
                lens_full[beta]["obs"].append(
                    float(np.sqrt(s_vec[-kappa:] @ s_vec[-kappa:] + e_vec @ e_vec))
                ) #estimated vs factual norms
            else:
                del bse_survivors[i]

            

        stats_full[beta][0] = len(bse_survivors)

        # 8) Full-dimension admissibility
        full_dim_succ = 0
        for b_vec, s_vec, e_vec in bse_survivors:
            sguess = s_vec[-kappa:]

            w1 = np.concatenate([sguess[:kappa // 2], np.zeros(len(sguess) - kappa // 2, dtype=int)])
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

        
        z_shape = [bkzgsa_gso_len(m*log(q),i, n+m-kappa,beta)**2 for i in range(n+m-kappa)]

        # z_shape = CN11( n+m-kappa,n-kappa,q,beta )

        r_vec = [G.get_r(i,i) for i in range(n+m-kappa)]
        bdd_err_norm = expected_bdd_err_norm(n+m-kappa, dist_e, dist_s,  dist_param_s, dist_param_e)


        stats_full[beta][1] = full_dim_succ
        stats_full[beta][2] += adm_probability2(n+m-kappa, r_vec, bdd_err_norm)
        stats_full[beta][3] += adm_probability2(n+m-kappa, z_shape, bdd_err_norm)


        # 9) Projected admissibility by cd
        for cd in cds:
            cd = int(cd)
            cd_dim_succ = 0

            for b_vec, s_vec, e_vec in bse_survivors:
                sguess = s_vec[-kappa:]

                w1 = np.concatenate([sguess[:kappa // 2], np.zeros(len(sguess) - kappa // 2, dtype=int)])
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

                v = np.concatenate([-e_vec, s_vec])[:-kappa]
                obs_proj_norm = gs_projected_canonical_norm(G, v, cd)
                lens_proj[beta][cd]["obs"].append(obs_proj_norm)

            bdd_err_norm_proj = lens_proj[beta][cd]["pred"]
            stats_proj[beta][cd][0] = cd_dim_succ
            stats_proj[beta][cd][1] += adm_probability2(cd, r_vec[-cd:], bdd_err_norm_proj)
            stats_proj[beta][cd][2] += adm_probability2(cd, z_shape[-cd:], bdd_err_norm_proj)


        elapsed_s = time.time() - t0
        #print("beta = ", beta, " finished for exp_id = ", exp_id)


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
        "beta": beta,
        "results_fulldim": stats_full,
        "result_proj": stats_proj,
        "lens_full": lens_full,
        "lens_proj": lens_proj,
    }

    # 11) Dump per-experiment pickle
    out_path = experiments_dir / f"exp_fp_{exp_id:04d}_{n}_{q}_{kappa}_{dist_s}_{dist_param_s}_{dist_e}_{dist_param_e}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(experiment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("lattice #", exp_id, " finished")
    

    return experiment_dict


def main():
    print(f"Running {n_lattices} independent lattices with {max_workers} workers.")
    print(f"Results will be written to: {experiments_dir.resolve()}")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_lattice, exp_id, beta_values) for exp_id in range(n_lattices)]

        for fut in as_completed(futures):
            res = fut.result()
            all_results.append(res)

            # print(
            #     f"[exp {res['exp_id']}] beta={res['beta']}, "
            #     f"babai_lift_success={res['babai_lift_success']}, "
            #     f"full_dim_succ={res['full_dim_succ']}"
            # )
            # print(res["results"], flush=True)


    #print("All results:")
    #print(all_results)


    #combine stats from all lattices and normalize
    stats_full = dict(  [ (beta, [0, 0, 0, 0] ) for beta in beta_values]  )
    stats_proj = dict( [ (beta, dict([ (int(cd), [0, 0, 0]) for cd in cds ])) for beta in beta_values] )
    lens_full_all = {
        beta: {"pred": None, "obs": []}
        for beta in beta_values
    }
    lens_proj_all = {
        beta: {
            int(cd): {"pred": None, "obs": []}
            for cd in cds
        }
        for beta in beta_values
    }

    for res in all_results:
        for beta, l in res["results_fulldim"].items():
            stats_full[beta][0] += int(l[0])
            stats_full[beta][1] += int(l[1])
            stats_full[beta][2] += float(l[2])
            stats_full[beta][3] += float(l[3])
        for beta in res["result_proj"].keys():
            for cd, l in res["result_proj"][beta].items():
                stats_proj[beta][cd][0] += int(l[0])
                stats_proj[beta][cd][1] += float(l[1])
                stats_proj[beta][cd][2] += float(l[2])
        for beta, d in res["lens_full"].items():
            lens_full_all[beta]["pred"] = float(d["pred"])
            lens_full_all[beta]["obs"].extend(float(x) for x in d["obs"])

        for beta, dd in res["lens_proj"].items():
            for cd, d in dd.items():
                lens_proj_all[beta][cd]["pred"] = float(d["pred"])
                lens_proj_all[beta][cd]["obs"].extend(float(x) for x in d["obs"])

    # take average
    for beta in stats_full.keys():
        for i in range(len(stats_full[beta])):
            stats_full[beta][i]/=n_lattices
    for beta in stats_proj.keys():
        for cd in stats_proj[beta].keys():
            for i in range(len(stats_proj[beta][cd])):
                stats_proj[beta][cd][i]/=n_lattices

    print("stats_full:")
    print(stats_full)

    print("stats_proj:")
    print(stats_proj)

    print("lens_full:")
    print(lens_full_all)

    print("lens_proj:")
    print(lens_proj_all)

    lens_full_summary = {
        beta: summarize_prediction(d["pred"], d["obs"])
        for beta, d in lens_full_all.items()
    }

    lens_proj_summary = {
        beta: {
            cd: summarize_prediction(d["pred"], d["obs"])
            for cd, d in dd.items()
        }
        for beta, dd in lens_proj_all.items()
    }

    print("lens_full_summary:")
    print(lens_full_summary)

    print("lens_proj_summary:")
    print(lens_proj_summary)

    # Optional combined dump
    combined = {
        "n_lattices": int(n_lattices),
        "all_results": all_results,
        "stats_full_avg": stats_full,
        "stats_proj_avg": stats_proj,
        "lens_full": lens_full_all,
        "lens_proj": lens_proj_all,
        "lens_full_summary": lens_full_summary,
        "lens_proj_summary": lens_proj_summary,
    }
    combined_path = experiments_dir / "all_experiments_fp.pkl"
    with open(combined_path, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved combined results to {combined_path.resolve()}")


if __name__ == "__main__":
    main()