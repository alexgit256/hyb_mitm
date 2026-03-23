from lwe_gen import generateLWEInstances
from lattice_reduction import LatticeReduction
import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, FPLLL
from fpylll.util import gaussian_heuristic
FPLLL.set_precision(80)

import time, random
from time import perf_counter
from random import randrange, shuffle
import pickle
import sys, os 

from utils import *
from math import log,e, pi

from concurrent.futures import ThreadPoolExecutor, as_completed
from threadpoolctl import threadpool_limits
THREADPOOL_LIMIT = 2 #set to 0 to allow max number of workers
from typing import Dict, Any, Optional
import concurrent.futures
from dataclasses import dataclass

import numpy as np

import traceback

# - - - BEG BLASTER SRed - - -

from blaster_core import \
    set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, ZZ_right_matmul
from size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce, nearest_plane
from stats import get_profile, rhf, slope, potential
from lattice_io import read_qary_lattice

from random import randrange

from copy import copy, deepcopy

# - - - END BLASTER SRed - - -

in_path = "./lwe_instances/saved_lattices/"
exp_path = "./exp_dir/"

from batch_attack_instance import BatchAttackInstance

def check_pairs_guess_admis(
        BAI:BatchAttackInstance, 
        start=0, end=None, 
        correct=True, 
        n_trials=10, 
        n_workers=1, 
        num_per_batch=512,           
        ):
    """
    Constructs correct (if correct==True) guesses s=s1-s2 for the MitM attack on LWE and checks if Babai_{H}(t+g2) == Babai(g1),
    where gi = si * C^{-1} (row notation)
    n_trials: number of decompositions of s.
    n_workers: number of parallel checks (threads).
    num_per_batch: batch size for babai
    Returns:
        minddinfs : list of per-(b,s,e) minimal infinity norms found across trials
    """
    if end is None:
        end = len(BAI.bse)
    np.set_printoptions(precision=4, suppress=True)
    # Helper that performs a single trial for current (b, s_correct_guess)
    def _trial_worker_admis(trie_idx, batch_size,b,s,e):
        with threadpool_limits(limits=THREADPOOL_LIMIT):
            # create a local RNG to avoid shared-state race
            seed = (os.getpid() ^ trie_idx ^ int(time.time_ns()))
            rng = np.random.default_rng(seed)
            kappa = BAI.kappa
            
            if not correct:  
                raise NotImplementedError
                # sguess = rng.integers(-1, 2, size=(kappa,batch_size), dtype=np.int16)
            else:
                s_corr = np.asarray(s[-BAI.kappa:], dtype=np.int64)
                msk_sublen = rng.integers(BAI.kappa//2, BAI.kappa, size=batch_size)
                msk = np.zeros((kappa, batch_size), dtype=np.int8)
                for j in range(batch_size):
                    msk[:msk_sublen[j], j] = 1
                    rng.shuffle(msk[:, j])
                sguess_1 = msk * s_corr[:, None]
                # sguess_2 = s_corr[:, None] - sguess_1
                #now w = sguess_1 + sguess_2


            Cvg = BAI.QinvC@s_corr[:, None] #C vg  (note: NP(Cvg) should be v_l)
            Cvg = Cvg[-BAI.cd:,:] #go to the projective sublattice

            Cw = BAI.QinvC@sguess_1  #C w
            Cw = Cw[-BAI.cd:,:] #go to the projective sublattice

            # computing NPCvg (debug)
            NPCvg = copy( Cvg )
            W_1 = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], NPCvg, dim=BAI.cd )
            # print(Cvg - NPCvg)
            # print()

            # computing NP( Cw )
            NPCw = copy( Cw )
            W_1 = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], NPCw, dim=BAI.cd )
            NPCw = NPCw #+ BAI.R[-BAI.cd:,-BAI.cd:]@W_1 #NP( C v_g )

            # computing NP( C v_g - Cw )
            NPCvg_Cw = Cvg - Cw
            W_2 = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], NPCvg_Cw, dim=BAI.cd )

            overall_dim = BAI.n+BAI.m-BAI.kappa
            true_err = BAI.Qinv @ np.concatenate([e,-s[:-BAI.kappa]]).T #v_l in Son Cheon 2019/1019.pdf (rotated to align with R-factor)
            proj_true_err = BAI.vec_project_onto( true_err, start=overall_dim-BAI.cd, end=overall_dim ) #projection of v_l (needed for the correct equation)

            #  - - - Checking p_{NP} in Lemma 4.2 proof - - - 
            pte = proj_true_err[-BAI.cd:]
            pte_cp = np.atleast_2d(pte).T
            proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], pte_cp, dim=BAI.cd )
            # print()
            tmp = pte - pte_cp[:, 0]
            mask_NP = np.all(np.abs(tmp) <= 1e-6, axis=0)
            print(f"any good_col_NP: {np.any(mask_NP)}")
            #  - - - END Checking p_{NP} in Lemma 4.2 proof - - - 

            #  - - - Checking p_s in Lemma 4.2 proof - - - 
            Rdiag = np.diag( BAI.R[-BAI.cd:,-BAI.cd:] )

            tmp = NPCw - NPCvg #should be in fund. par. by the proof of Lemma 4.2 (https://eprint.iacr.org/2019/1019.pdf)
            tmp_scaled = tmp / Rdiag[:, None]
            mask_s = np.all(np.abs(tmp_scaled) <= 0.5, axis=0)
            # print( tmp_scaled )
            print( np.any(mask_s) )
            good_cols_s = np.where(mask_s)[0]
            print(f"any good_col_s forward: {good_cols_s}")
            print()

            tmp = NPCw + NPCvg_Cw #should be in fund. par. by the proof of Lemma 4.2 (https://eprint.iacr.org/2019/1019.pdf)
            tmp_scaled = tmp / Rdiag[:, None]
            mask_s = np.all(np.abs(tmp_scaled) <= 0.5, axis=0)
            # print( tmp_scaled )
            print( np.any(mask_s) )
            good_cols_s = np.where(mask_s)[0]
            print(f"any good_col_s backward: {good_cols_s}")
            
            idx = good_cols_s[0]
            lhs = tmp[:,idx:idx+1]
            proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], lhs, dim=BAI.cd ) #because we do CVP, not SVP as in the paper
            print(f"lhs = NPCvg: {np.all(np.abs(lhs - NPCvg) <= 1e-6)}")
            # print(lhs - NPCvg)

            # - - - END Checking p_s in Lemma 4.2 proof - - - 

            """
            print("- - - - sanity check - - -")
            
            pte = proj_true_err[-BAI.cd:]
            pte_cp = np.atleast_2d(pte).T
            proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], pte_cp, dim=BAI.cd )
            print(pte - pte_cp[:, 0])
            print()

            print("end test - - - -")
            """

            NPCw_Cvg = Cw - Cvg
            W_2 = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], NPCw_Cvg, dim=BAI.cd )

            # checking if NP( C v_g - Cw ) + NP( Cw ) = v_l which becomes
            # NP( Cw - Cvg ) = v_l in our notation (!)
            is_adm = []
            tmp12 = (NPCw_Cvg-NPCvg)

            for col in tmp12.T:
            # for j in good_cols_s:
            #     col = tmp12[:, j]
                ccol = col[-BAI.cd:] - proj_true_err[-BAI.cd:]
                # in our notations it means ccol is in proj. lat.
                proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], np.atleast_2d(ccol).T, dim=BAI.cd )
                print(f"ccol: {ccol}")
                okay = all(np.isclose(ccol,0.0,atol=0.001))
                is_adm.append(okay)


        return sum(is_adm) 

    # For each (b,s,e) in bse, run n_trials of _trial_worker_dist (parallelised)
    is_adm_num = 0
    is_adm_nums = []
    target_num = 0
    for cntr in range(start, end):  #iterate only through successfull babai's
        if cntr in BAI.correct_indices:
            b, s, e = BAI.bse[cntr]
            n_trials_normalized = (n_trials)//num_per_batch #num_per_batch used to be one
            if n_workers is None or n_workers <= 1:
                # sequential fallback
                for tries in range(n_trials_normalized):
                    if tries != 0 and tries % 10 == 0:
                        print(f"{tries} out of {n_trials_normalized} done")
                    is_adm = _trial_worker_admis(tries, num_per_batch,b,s,e)
                    is_adm_num+=is_adm
            else:
                # parallel execution using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    fut_kind = {}  # future -> ("dist"|"admis", tries)
                    for tries in range(n_trials_normalized):
                        fut = ex.submit(_trial_worker_admis, tries, num_per_batch, b, s, e)
                        fut_kind[fut] = (tries)

                    # collect
                    for fut in as_completed(fut_kind):
                        tries = fut_kind[fut]

                        if tries != 0 and tries % 10 == 0:
                            print(f"{tries} out of {n_trials_normalized} done")

                        try:
                            res = fut.result()
                        except Exception:
                            print(f"\n_trial_worker crashed for tries={tries}")
                            traceback.print_exc()
                            continue
                        # _trial_worker_admis returns sum(is_adm)
                        is_adm_num += int(res)
            target_num += 1
            # print(f"is_adm_num: {is_adm_num} | {n_trials_normalized*num_per_batch}")
            if target_num%10==0:
                print(f"{target_num} out of {end-start} targets done", flush=True)
            
            is_adm_nums.append(is_adm_num)
            is_adm_num = 0
    return is_adm_nums


# NEW: worker to run one complete BatchAttackInstance experiment in a separate process
def run_single_instance(idx: int,
                    n: int, m: int, q: int, n_tars: int,
                    dist_s: str, dist_param_s: int,
                    dist_e: str, dist_param_e: int,
                    kappa: int, cd: int,
                    beta_max: int,
                    seed_base: int,
                    n_trials: int,
                    num_per_batch: int,
                    ntar: int,
                    inner_n_workers: int,
                    babai_succ_rate=0.95) -> Dict[str, Any]:
    """
    One worker:
        - seeds RNG
        - loads or creates + reduces a BatchAttackInstance
        - runs check_correct_guess and check_pairs_guess_admis (correct/incorrect)
        - returns summary dict
    """
    # per-instance seed (for reproducibility and diversity)
    seed = seed_base + idx
    random.seed(seed)
    np.random.seed(seed)

    # filename unique per instance
    filename = f"lweinst_{n}_{m}_{q}_{dist_s}_{dist_param_s:.4f}_{dist_e}_{dist_param_e:.4f}_{kappa}_inst{idx}.pkl"
    fullpath = os.path.join(in_path, filename)

    # load or create instance
    try:
        lwe_instance = BatchAttackInstance.load_from_disc(fullpath)
        lwe_instance.cd = cd #technically, this should not be a field?
        loaded = True
        print(f"[inst {idx}] loaded from {fullpath}")
    except FileNotFoundError:
        loaded = False
        print(f"[inst {idx}] creating new instance (will save to {fullpath})")
        lwe_instance = BatchAttackInstance(
            n=n, m=m, q=q, n_tars=n_tars,
            dist_s=dist_s, dist_param_s=dist_param_s,
            dist_e=dist_e, dist_param_e=dist_param_e,
            kappa=kappa, cd=cd, ntar=ntar
        )

        if babai_succ_rate:
            # vefifies
            nrms = []
            for _, s, e in lwe_instance.bse:
                v = np.concatenate([s,e])
                nrms.append( v@v )
            #estimation of projected length (95th percentile)
            nrmper = np.percentile( nrms, 0.95 ) * ( lwe_instance.cd / (n+m) )**2

        for beta in list(range(40, beta_max+1, 1)):
            t0 = perf_counter()
            lwe_instance.reduce(beta=beta, bkz_tours=2, 
                                cores=inner_n_workers, #, depth=4,
                                start=0, end=None)
            print(f"[inst {idx}] bkz-{beta} done in {perf_counter()-t0:.2f}s")

            if babai_succ_rate:
                succ, iters = lwe_instance.check_correct_guesses()
                if succ/iters >= babai_succ_rate:
                    print(f"Stopped lattice reduction [{idx}] at {beta}: {succ/iters} >= {babai_succ_rate}")
                    break
                else:
                    print(f"Continuing reduction [{idx}]: {beta}: {succ/iters} < {babai_succ_rate}")
            

        # save reduced instance to disk so future runs can reuse it
        os.makedirs(in_path, exist_ok=True)
        lwe_instance.dump_on_disc(fullpath)
        print(f"[inst {idx}] dumped to {fullpath}")

    lwe_instance = BatchAttackInstance.load_from_disc(fullpath)
    lwe_instance.cd = cd #technically, this should not be a field?
    loaded = True
    print(f"[inst {idx}] loaded from {fullpath}")


    # run experiments
    print(f"[inst {idx}] check_correct_guesses()")
    succnum, itnum = lwe_instance.check_correct_guesses()
    print(f"[inst {idx}] check_correct_guess -> ({succnum}, {itnum})")

    is_adm_num_incorrect = None
    # print(f"[inst {idx}] check_pairs_guess_admis(correct=False)")
    # is_adm_num_incorrect = check_pairs_guess_admis(
    #     lwe_instance,n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=False)

    print(f"[inst {idx}] check_pairs_guess_admis(correct=True)")
    is_adm_num_correct   = check_pairs_guess_admis(
        lwe_instance, n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=True)

    print(f"[inst {idx}] correct adm: {is_adm_num_correct}")
    print(f"[inst {idx}] incorrect adm: {is_adm_num_incorrect}")

    return {
        "version": "0.0.3a",  #file format version
        "idx": idx,
        "seed": seed,
        'n': n,
        'm': m,
        'q': q,
        'beta_max': beta_max,
        'kappa': kappa,
        'cd': cd,
        'dists': {
            "dist_s": dist_s, "dist_param_s": dist_param_s,
            "dist_e": dist_e, "dist_param_e": dist_param_e,
        },
        'n_trials': n_trials,
        "filename": filename,
        "is_adm_num_correct": is_adm_num_correct,
        "is_adm_num_incorrect": is_adm_num_incorrect,
        "succnum": succnum,
        "itnum": itnum,
        "loaded": loaded,
    }
        
def main():
    # outer parallelism: number of independent BatchAttackInstance runs
    n_workers = 2  # set this >1 to parallelize across instances
    n_lats = 2  # number of lattices    #5
    n_tars = 20 ## per-lattice instances #20
    n_trials = 256*4 #256*4          # per-lattice-instance trials in check_pairs_guess_admis. SHOULD be div'e by num_per_batch
    num_per_batch = 256 #256 #do not go above 256
    inner_n_workers = 2   # threads for inner parallelism

    assert n_trials%num_per_batch == 0, f"n_trials should be divisible by num_per_batch"

    n, m, q = 112, 112, 3329
    seed_base = 0
    dist_s, dist_param_s, dist_e, dist_param_e = "binomial", 2, "ternary_sparse", 32
    kappa = 10
    cd = 30
    beta_max = 45
    babai_succ_rate = 1.01

    os.makedirs(in_path, exist_ok=True)

    results = []
    # use processes for independent instances (CPU-bound)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                run_single_instance,
                idx,
                n, m, q, n_tars,
                dist_s, dist_param_s,
                dist_e, dist_param_e,
                kappa, cd,
                beta_max,
                seed_base,
                n_trials,
                num_per_batch,
                n_tars,
                inner_n_workers,
                babai_succ_rate,
            ): idx
            for idx in range(n_lats)
        }

        try:
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[main] instance {idx} raised {e!r}")
                    res = {"idx": idx, "error": repr(e)}
                    raise e
                results.append(res)
        except KeyboardInterrupt:
            print("\n[main] Caught KeyboardInterrupt, cancelling workers...")
            # cancel all pending futures
            for f in futures:
                f.cancel()
            # tell the executor to stop without waiting for the workers
            executor.shutdown(wait=False, cancel_futures=True)
            # re-raise so outer handler can catch it
            raise

    # compact summary
    # for r in results:
    #     if "error" in r:
    #         print(f"instance {r['idx']} ERROR: {r['error']}")
    #     else:
    #         print(f"instance {r['idx']} (seed {r['seed']}, file {r['filename']}, loaded={r['loaded']})")
    #         print(f"  check_correct_guess: ({r['succnum']}, {r['itnum']})")

    return results

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    os.makedirs(exp_path, exist_ok=True)
    try:
        results = main()
        now = get_current_datetime()
        with open(exp_path+f"res_admis_{now}.pkl","wb") as file:
            pickle.dump( results, file )
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(1)

    # compact summary
    for r in results:
        if "error" in r:
            print(f"instance {r['idx']} ERROR: {r['error']}")
        else:
            print(f"instance {r['idx']} (seed {r['seed']}, file {r['filename']}, loaded={r['loaded']})")
            print(f"  check_correct_guess: ({r['succnum']}, {r['itnum']})")