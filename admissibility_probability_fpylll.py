import time, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from math import sqrt, log

import numpy as np
from fpylll import IntegerMatrix, GSO, FPLLL

from lwe_gen import generateLWEInstances

from zgsa_fast import bkzgsa_gso_len, adm_probability2, CN11, mitm_babai_probability
from admissibility_helpers import (
    project_onto_last, gs_projected_canonical_norm, split_secret_guess, 
    split_secret_guess, lwe_target, babai_residual, filter_babai_lift_survivors,
    summarize_prediction, expected_bdd_err_norm, compute_beta,
    build_partitioned_basis, reduce_lattice, fill_zero_projected_results
)
from math import sqrt, log
from PT25 import expected_proj_norm
from time import perf_counter


# - - - Full dim loop replacement - - -
def check_full_admissibility(G, b_vec, s_vec, e_vec, C, n, kappa, atol=1e-7):
    sguess = s_vec[-kappa:]
    w1, w2 = split_secret_guess(sguess, kappa)

    target_w1 = lwe_target(b_vec, w1, C, n, kappa)
    target_w2 = -w2 @ C

    err_w1_gs = np.asarray(G.from_canonical(babai_residual(G, target_w1)), dtype=float)
    err_w2_gs = np.asarray(G.from_canonical(babai_residual(G, target_w2)), dtype=float)

    true_err_gs = np.asarray(
        G.from_canonical(np.concatenate([e_vec, -s_vec[:-kappa]])),
        dtype=float,
    )

    return np.all(np.isclose(true_err_gs - err_w1_gs - err_w2_gs, 0.0, atol=atol))

def compute_full_dimension_admissibility(
    G,
    bse_survivors,
    C,
    n,
    m,
    q,
    kappa,
    beta,
    dist_e,
    dist_s,
    dist_param_s,
    dist_param_e,
    atol=1e-7,
):
    d = n + m - kappa

    full_dim_succ = sum(
        check_full_admissibility(G, b_vec, s_vec, e_vec, C, n, kappa, atol=atol)
        for b_vec, s_vec, e_vec in bse_survivors
    )

    r_vec = [G.get_r(i, i) for i in range(d)]
    z_shape = [bkzgsa_gso_len(m * log(q), i, d, beta) ** 2 for i in range(d)]
    bdd_err_norm = expected_bdd_err_norm(
        d, dist_e, dist_s, dist_param_s, dist_param_e
    )

    return {
        "successes": full_dim_succ,
        "prob_exact_r": adm_probability2(d, r_vec, bdd_err_norm),
        "prob_gsa": adm_probability2(d, z_shape, bdd_err_norm),
        "prob_mitm_babai": mitm_babai_probability(r_vec, bdd_err_norm / sqrt(d)),
        "r_vec": r_vec,
        "z_shape": z_shape,
        "bdd_err_norm": bdd_err_norm,
    }

# - - - Projected dim loop replecement - - -

def check_projected_admissibility(
    G, b_vec, s_vec, e_vec, C, n, kappa, cd, atol=1e-7
):
    sguess = s_vec[-kappa:]
    w1, w2 = split_secret_guess(sguess, kappa)

    target_w1 = lwe_target(b_vec, w1, C, n, kappa)
    target_w2 = -w2 @ C

    target_w1_proj = project_onto_last(G, target_w1, cd)
    target_w2_proj = project_onto_last(G, target_w2, cd)

    err_w1_proj_gs = np.asarray(
        G.from_canonical(babai_residual(G, target_w1_proj)),
        dtype=float,
    )[-cd:]

    err_w2_proj_gs = np.asarray(
        G.from_canonical(babai_residual(G, target_w2_proj)),
        dtype=float,
    )[-cd:]

    true_err_proj_gs = np.asarray(
        G.from_canonical(np.concatenate([e_vec, -s_vec[:-kappa]])),
        dtype=float,
    )[-cd:]

    return np.all(
        np.isclose(true_err_proj_gs - err_w1_proj_gs - err_w2_proj_gs, 0.0, atol=atol)
    )


def compute_projected_admissibility_by_cd(
    G,
    bse_survivors,
    C,
    n,
    kappa,
    cds,
    r_vec,
    z_shape,
    lens_proj_beta,
    atol=1e-7,
):
    result = {}

    cds_sorted = sorted(map(int, cds))

    # All vectors that survived the full Babai-lift stage.
    # These are the denominator / norm-sample population.
    all_survivors = list(bse_survivors)

    # Active vectors are only for admissibility checking.
    # Once a vector fails at cd0, it need not be Babai-checked for larger cd.
    active = list(bse_survivors)

    for cd in cds_sorted:
        # Collect observed projected norms for all Babai-lift survivors,
        # not only for projected-admissibility successes.
        obs_norms = []
        for b_vec, s_vec, e_vec in all_survivors:
            v = np.concatenate([-e_vec, s_vec])[:-kappa]
            obs_norms.append(gs_projected_canonical_norm(G, v, cd))

        next_active = []

        for b_vec, s_vec, e_vec in active:
            if check_projected_admissibility(
                G, b_vec, s_vec, e_vec, C, n, kappa, cd, atol=atol
            ):
                next_active.append((b_vec, s_vec, e_vec))

        bdd_err_norm_proj = lens_proj_beta[cd]["pred"]

        result[cd] = {
            "successes": len(next_active),
            "checked": len(active),
            "norm_count": len(obs_norms),
            "prob_exact_r": adm_probability2(cd, r_vec[-cd:], bdd_err_norm_proj),
            "prob_gsa": adm_probability2(cd, z_shape[-cd:], bdd_err_norm_proj),
            "prob_mitm_babai": mitm_babai_probability(
                r_vec[-cd:], bdd_err_norm_proj / sqrt(cd)
            ),
            "obs_norms": obs_norms,
        }

        active = next_active

    return result

# ----------------------------
# Configuration
# ----------------------------
FPLLL.set_precision(208)
# Parallelism over lattices
max_workers = 8 #min(n_lattices, os.cpu_count() or 1)

n, m, q = 100, 100, 3329
dist_s, dist_param_s = "binary", 0.5
dist_e, dist_param_e = "binary", 0.5

kappa = 25
# Number of independent lattices / experiments
n_lattices = 4
n_targets = 2000

a, b, n_dims = 30, min(100, n + m - kappa), 8
cds = np.asarray(np.round(np.linspace(a, b, n_dims)), dtype=int)
# cds = [50,75]
print("cd values:", cds)

bkz_tours = 5
lll_size = 64
# Compute beta
beta_s = compute_beta(n, m, q, kappa, dist_e, dist_param_e, cds[0])+10
BETA_HARD_CAP = 80
beta_values = [beta_s+i*10 for i in range(4) if beta_s+i*10<BETA_HARD_CAP]

print("beta values:", beta_values)



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
        n_targets, seed=exp_id,
    )
    assert len(bse) == n_targets

    Hred, C =  build_partitioned_basis(A, n, m, q, kappa)

    # dictionary to collect statistic on full lattice
    # [
    # # babai success on full dim, 
    # #succ admissibility on full dim, 
    # estimated adm. succ using exact R, 
    # estimated adm. succ using GSA
    # estimated adm. succ using exact R (Ludo Pulles' code)
    # ]
    stats_full = dict(  [ (beta, [0, 0, 0, 0, 0] ) for beta in beta_values]  )

    # dictionary to collect statistic on projected lattices
    #for each beta and each cd, collect the same data as for stats_full except # babai success on full dim
    stats_proj = dict( [ (beta, dict([ (int(cd), [0, 0, 0, 0]) for cd in cds ])) for beta in beta_values] )

    print(f"- - - {(n + m - kappa, dist_e, dist_s, dist_param_s, dist_param_e)} - - -")
    print(expected_bdd_err_norm(n + m - kappa, dist_e, dist_s, dist_param_s, dist_param_e))

    lens_full = {
        beta: {
            "pred": expected_bdd_err_norm(n + m - kappa, dist_e, dist_s, dist_param_s, dist_param_e), #2025/2195 is irrelevant w/o projection
            "obs": [],
        }
        for beta in beta_values
    }

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
        Hred = reduce_lattice(Hred, beta, lll_size, bkz_tours, cores=2)

        # 6) Build GSO
        G = GSO.Mat(IntegerMatrix.from_matrix(Hred), float_type="mpfr")
        G.update_gso()

        # 7) Babai-lift filtering
        bse_survivors = list(bse)
        babai_lift_success = 0

        bse_survivors, babai_lift_success, obs_norms = filter_babai_lift_survivors(
            G, bse, C, n, kappa
        )

        stats_full[beta][0] = babai_lift_success
        lens_full[beta]["obs"].extend(obs_norms)

        # print(f" - - - b:{beta} | bls:{babai_lift_success} ")   

        full = compute_full_dimension_admissibility(
            G, bse_survivors, C, n, m, q, kappa, beta,
            dist_e, dist_s, dist_param_s, dist_param_e,
        )

        stats_full[beta][1] = full["successes"]
        stats_full[beta][2] += full["prob_exact_r"]
        stats_full[beta][3] += full["prob_gsa"]
        stats_full[beta][4] += full["prob_mitm_babai"]

        proj = compute_projected_admissibility_by_cd(
        G,
        bse_survivors,
        C,
        n,
        kappa,
        cds,
        full["r_vec"],
        full["z_shape"],
        lens_proj[beta],
        )

        for cd, row in proj.items():
            stats_proj[beta][cd][0] = row["successes"]
            stats_proj[beta][cd][1] += row["prob_exact_r"]
            stats_proj[beta][cd][2] += row["prob_gsa"]
            stats_proj[beta][cd][3] += row["prob_mitm_babai"]
            lens_proj[beta][cd]["obs"].extend(row["obs_norms"])


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


    #combine stats from all lattices and normalize
    stats_full = dict(  [ (beta, [0, 0, 0, 0] ) for beta in beta_values]  )
    stats_proj = dict( [ (beta, dict([ (int(cd), [0, 0, 0, 0]) for cd in cds ])) for beta in beta_values] )
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
                stats_proj[beta][cd][3] += float(l[3])
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
    print( f"Observed: {np.mean( lens_full_all[beta]['obs'] )}" )
    print( f"Predicted:{np.mean( lens_full_all[beta]['pred'] )}" )
    # print(lens_full_all)

    print("lens_proj:")
    for cd in cds:
        print(f"pred: {lens_proj_all[beta][cd]["pred"]}")
        print(f"pred-cd-{cd}: {np.mean(lens_proj_all[beta][cd]["obs"])}")
    # print(lens_proj_all)

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