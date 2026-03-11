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
    
    # Helper that performs a single trial for current (b, s_correct_guess)
    def _trial_worker_admis(trie_idx, batch_size,s,e):
        with threadpool_limits(limits=THREADPOOL_LIMIT):
            # create a local RNG to avoid shared-state race
            seed = (os.getpid() ^ trie_idx ^ int(time.time_ns()))
            rng = np.random.default_rng(seed)
            kappa = BAI.kappa

            sguess = rng.integers(-1, 2, size=(kappa,batch_size), dtype=np.int16)
            sec_proj_cols = BAI.QinvCT@sguess #t2
            sec_proj_cols = sec_proj_cols[-BAI.cd:] #go to the projective sublattice

            # - - - Admissibility: check that babai(guess2 + err) = babai(guess2) + babai(err) - - -

            true_err = np.concatenate([e,-s[:-BAI.kappa]])[-BAI.cd:]
            tmp0batch = sec_proj_cols + true_err[:,None]
            W = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], tmp0batch, dim=BAI.cd )
            tmp0batch = tmp0batch + BAI.R[-BAI.cd:,-BAI.cd:]@W

            tmp1batch = sec_proj_cols
            W = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], tmp1batch, dim=BAI.cd )
            tmp1batch = tmp1batch + BAI.R[-BAI.cd:,-BAI.cd:]@W

            true_err = np.atleast_2d(true_err).T.astype(np.float64)
            W = proj_submatrix_modulus_blas( BAI.R[-BAI.cd:,-BAI.cd:], true_err, dim=BAI.cd )
            tmp2 = true_err + BAI.R[-BAI.cd:,-BAI.cd:]@W

            is_adm = []
            tmp12 = tmp0batch - (tmp1batch+tmp2)
            for col in tmp12.T:
                okay = all(np.isclose(col,0.0,atol=0.001))
                is_adm.append(okay)

        return sum(is_adm) 

    # For each (b,s,e) in bse, run n_trials of _trial_worker_dist (parallelised)
    is_adm_num = 0
    is_adm_nums = []
    target_num = 0
    for b, s, e in BAI.bse[start:end]:
        n_trials_normalized = (n_trials)//num_per_batch #num_per_batch used to be one
        if n_workers is None or n_workers <= 1:
            # sequential fallback
            for tries in range(n_trials_normalized):
                if tries != 0 and tries % 10 == 0:
                    print(f"{tries} out of {n_trials_normalized} done")
                is_adm = _trial_worker_admis(tries, num_per_batch,s,e)
                is_adm_num+=is_adm
        else:
            # parallel execution using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                fut_kind = {}  # future -> ("dist"|"admis", tries)
                for tries in range(n_trials_normalized):
                    fut = ex.submit(_trial_worker_admis, tries, num_per_batch, s, e)
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

    print(f"[inst {idx}] check_pairs_guess_admis(correct=False)")
    is_adm_num_incorrect = check_pairs_guess_admis(
        lwe_instance,n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=False)

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
        # "infdiff_correct": infdiff_correct,
        # "infdiff_incorrect": infdiff_incorrect,
        # "mindds_correct": mindds_correct,
        # "mindds_incorrect": mindds_incorrect,
        "loaded": loaded,
    }
        
def main():
    # outer parallelism: number of independent BatchAttackInstance runs
    n_workers = 2  # set this >1 to parallelize across instances
    n_lats = 2  # number of lattices    #5
    n_tars = 20 ## per-lattice instances #20
    n_trials = 256*10 #256*4          # per-lattice-instance trials in check_pairs_guess_admis. SHOULD be div'e by num_per_batch
    num_per_batch = 256 #256 #do not go above 256
    inner_n_workers = 2   # threads for inner parallelism

    assert n_trials%num_per_batch == 0, f"n_trials should be divisible by num_per_batch"

    n, m, q = 130, 130, 4096
    seed_base = 0
    dist_s, dist_param_s, dist_e, dist_param_e = "binomial", 2, "gaussian", 1.5
    kappa = 25
    cd = 64
    beta_max = 45
    babai_succ_rate = 0.5

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