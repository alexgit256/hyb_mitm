USE_MASK = True

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
import sys, os  # NEW

from utils import *
from math import log,e, pi

from concurrent.futures import ThreadPoolExecutor, as_completed
from threadpoolctl import threadpool_limits
THREADPOOL_LIMIT = 0 #set to 0 to allow max number of workers
from typing import Dict, Any, Optional
import concurrent.futures
from dataclasses import dataclass

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter

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

def get_current_datetime():
    """
    Returns the current local date and time as a formatted string.
    """
    try:
        # Get current local date and time
        now = datetime.now()
        
        # Format: YYYY-MM-DD HH:MM:SS
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        return formatted
    except Exception as e:
        return f"Error getting date and time: {e}"

# def add_pm1_noise(rng, sguess, kappa, batch_size, m_low=None, m_high=None, width=1, dtype=np.int8):
def add_signed_uniform_noise(rng, sguess, kappa, batch_size, w=2, m_low=None, m_high=None, dtype=None):
    """
    Add per-column sparse noise where each nonzero entry is uniform in
    {-w,...,-1, 1,...,w} (no zero).

    For each column j, choose m[j] distinct positions and add that noise there.
    """
    w = int(w)
    if w <= 0:
        raise ValueError("w must be >= 1")

    if m_low is None:  m_low  = kappa // 2
    if m_high is None: m_high = kappa
    m = rng.integers(m_low, m_high, size=batch_size)  # per-column sparsity
    mmax = int(m.max())

    # Choose positions without per-column shuffles:
    R = rng.random((kappa, batch_size), dtype=np.float32)
    idx = np.argpartition(R, kth=kappa - mmax, axis=0)[kappa - mmax:, :]  # (mmax, batch)

    # Values: magnitude in 1..w, sign in ±1
    mag = rng.integers(1, w + 1, size=(mmax, batch_size), dtype=np.int16)  # 1..w
    sgn = (rng.integers(0, 2, size=(mmax, batch_size), dtype=np.int16) * 2 - 1)  # ±1
    vals = mag * sgn  # uniform over ±1..±w

    # Scatter only the first m[j] entries in each column
    rows = np.arange(mmax)[:, None]
    keep = rows < m[None, :]  # (mmax, batch)

    idx_kept = idx[keep]  # 1D
    col_kept = np.broadcast_to(np.arange(batch_size), (mmax, batch_size))[keep]
    val_kept = vals[keep]

    # Pick dtype that won't overflow when added to sguess
    if dtype is None:
        # conservative: int16 unless sguess is already wider
        dtype = np.int16 if sguess.dtype.itemsize <= 2 else sguess.dtype

    noise = np.zeros((kappa, batch_size), dtype=dtype)
    noise[idx_kept, col_kept] = val_kept.astype(dtype, copy=False)

    return sguess + noise

class BatchAttackInstance:
    def __init__(self, **kwargs):
        self.A = None
        self.q = None
        self.bse = None

        self.LR = None
        self.n = kwargs.get("n") 
        self.m = kwargs.get("m") 
        self.q = kwargs.get("q")
        self.n_tars = kwargs.get("n_tars")  
        self.dist_s,self.dist_param_s,self.dist_e,self.dist_param_e = kwargs.get("dist_s"), kwargs.get("dist_param_s"), kwargs.get("dist_e") , kwargs.get("dist_param_e")

        A, bse = kwargs.get("A"), kwargs.get("bse")
        if A is None or bse is None:
            A, self.q, bse  = generateLWEInstances(self.n,self.m,self.q,self.dist_s,self.dist_param_s,self.dist_e,self.dist_param_e,ntar=self.n_tars)
        self.A, self.bse = A, bse

        self.kappa = kwargs.get("kappa") #guessing dim
        self.cd = kwargs.get("cd") #CVP dim

        assert self.m+self.n-(self.kappa+self.cd) >=0, f"Too large attack dimensions!"

        #extract relevant sublattices
        H, C = kwargs.get("H"), kwargs.get("C")
        if H is None or C is None:
            n = self.n
            m = self.m
            B = [ [int(0) for j in range(m+n)] for i in range(m+n) ]
            for i in range( m ):
                B[i][i] = int( self.q )
            for i in range(m, m+n):
                B[i][i] = 1
            for i in range(m, m+n):
                for j in range(n):
                    B[i][j] = int( self.A[i-m,j] )

            H = B[:len(B)-self.kappa] #the part of basis to be reduced
            H = IntegerMatrix.from_matrix( [ h11[:len(B)-self.kappa] for h11 in H  ] )
            C = np.array( [ b[:len(B)-self.kappa] for b in B[len(B)-self.kappa:] ] ) #the guessing matrix 
        self.H, self.C = H, C

        #QR enforces column matrices - so we transpose everything for BLASTER
        Qinv, R, QinvCT = kwargs.get("Qinv"), kwargs.get("R"), kwargs.get("QinvCT")
        if any( tmp is None for tmp in [Qinv, R, QinvCT] ):
            Q, R = np.linalg.qr(np.array(list(H),dtype=np.float64).transpose(), mode='reduced')
            Qinv =  np.linalg.inv( Q )
            QinvCT = Qinv@np.array(C).astype(np.float64).transpose()
            self.R = R
            self.Qinv = Qinv
            self.QinvCT = QinvCT

    def reduce(self,beta, bkz_tours, bkz_size=64, lll_size = 64, delta = 0.99, cores=1, debug= False,
        verbose= True, logfile= None, anim= None, depth = 4,
        use_seysen = False, **kwds):
        if self.LR is None:
            self.LR = LatticeReduction( self.H )

        self.H = self.LR(lll_size=lll_size, delta=delta, cores=cores, debug=debug,
            verbose=verbose, logfile=logfile, anim=anim, depth=depth,
            use_seysen=use_seysen, beta=beta, bkz_tours=bkz_tours, bkz_size=64)
        
    def dump_on_disc(self, filename):
        with open(filename,"wb") as file:
            D = {
                "n": self.n,
                "m": self.m,
                "q": self.q,
                "n_tars": self.n_tars,
                "dist_s": self.dist_s,
                "dist_param_s": self.dist_param_s,
                "dist_e": self.dist_e,
                "dist_param_e": self.dist_param_e,
                "A": self.A,
                "bse": self.bse,
                "kappa": self.kappa,
                "cd": self.cd,
                "H": self.H,
                "C": self.C,
            }
            pickle.dump( D,file )

    @staticmethod
    def load_from_disc(filename):
        with open(filename,"rb") as file:
            D = pickle.load(file)
        return BatchAttackInstance(
            n = D.get("n"),
            m = D.get("m"),
            q = D.get("q"),
            n_tars = D.get("n_tars"),
            dist_s = D.get("dist_s"),
            dist_param_s = D.get("dist_param_s"),
            dist_e = D.get("dist_e"),
            dist_param_e = D.get("dist_param_e"),
            A = D.get("A"),
            bse = D.get("bse"),
            kappa = D.get("kappa"),
            cd = D.get("cd"),
            H = D.get("H"),
            C = D.get("C"),
        )
        
    #checks if, after guessing self.kappa coordinates, babai solves instances [start:end]
    def check_correct_guess(self, start=0, end=None):
        if end is None:
            end = len(self.bse)
        G = GSO.Mat( IntegerMatrix.from_matrix( self.H ), float_type="mpfr" )
        G.update_gso()
        succnum=0
        itnum=0
        for b, s, e in self.bse[start:end]:
            itnum+=1
            s_correct_guess = s[-self.kappa:]
            sec_proj = ( s_correct_guess @ self.C )
            t = np.concatenate( [b,(self.n-self.kappa)*[0]] )
            t -= sec_proj
            bab = G.babai(t)

            tshift = t - G.B.multiply_left(bab)
            
            diff = tshift-np.concatenate([e,-s[:-self.kappa]])
            if all( np.isclose(diff, 0.0, atol=1e-10) ):
                succnum+=1
        return succnum, itnum

    #updates T inplace
        #updates T inplace
    def _apply_proj_submatrix_modulus(self,R,T,dim=None):
        T = T[-self.cd:]
        Tnew = copy(T)
        U2 = proj_submatrix_modulus_blas( R, Tnew, dim=dim )
        T += Tnew + self.R[-self.cd:,-self.cd:]@U2
        return T
                   
    def check_pairs_guess_dist(self, start=0, end=None, correct=True, n_trials=10, n_workers=1, num_per_batch=512):
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
            end = len(self.bse)
        G = GSO.Mat( IntegerMatrix.from_matrix( self.H ), float_type="mpfr" )
        G.update_gso()

        mindds, minddinfs = [], []

        
        # Helper that performs a single trial for current (b, s_correct_guess)
        def _trial_worker(trie_idx, batch_size, b, s_correct_guess,s,e):
            with threadpool_limits(limits=THREADPOOL_LIMIT):
                # create a local RNG to avoid shared-state race
                seed = (os.getpid() ^ trie_idx ^ int(time.time_ns()))
                rng = np.random.default_rng(seed)

                # ensure numpy arrays for elementwise ops
                s_corr = np.asarray(s_correct_guess, dtype=np.int64)
                kappa = s_corr.size

                # if correct:
                msk_sublen = rng.integers(self.kappa//2, self.kappa, size=batch_size)

                # build masks: shape (kappa, B)
                msk = np.zeros((kappa, batch_size), dtype=np.int8)
                for j in range(batch_size):
                    msk[:msk_sublen[j], j] = 1
                    rng.shuffle(msk[:, j])

                # broadcast s_corr[:,None]
                sguess_1 = msk * s_corr[:, None] 
                sguess_2 = -sguess_1 + s_corr[:, None]    
                if not correct: #simulate a "slight" misguess 
                    # sguess_1 = add_signed_uniform_noise(rng, sguess_1, kappa, batch_size, w=1, m_low=1, m_high=min(7,kappa//2))
                    # sguess_2 = add_signed_uniform_noise(rng, sguess_2, kappa, batch_size, w=1, m_low=1, m_high=min(7,kappa//2))
                    sguess_1 += rng.integers(-1, 2, size=(kappa, batch_size), dtype=np.int16)
                    sguess_2 += rng.integers(-1, 2, size=(kappa, batch_size), dtype=np.int16)

                # compute projections and shifts
                sec_proj1_cols = self.QinvCT@sguess_1  #note: sguess_1 is a guess, so we can use it here

                t1 = self.Qinv@np.concatenate( [b,(self.n-self.kappa)*[0]] ) #original target alligned wrt GS vectors
                tbatch1 = t1[:,None] - sec_proj1_cols   #t1 = target - guess_1
                tbatch1 = self._apply_proj_submatrix_modulus( self.R[-self.cd:,-self.cd:], tbatch1, dim=self.cd )

                sec_proj2_cols = self.QinvCT@sguess_2 #t2
                sec_proj2_cols = sec_proj2_cols[-self.cd:] #go to the projective sublattice
                tbatch2 = copy(sec_proj2_cols) #we will still need sec_proj2_cols intact
                tbatch2 = self._apply_proj_submatrix_modulus( self.R[-self.cd:,-self.cd:], tbatch2, dim=self.cd )
                

                dbatch = tbatch1 - tbatch2 #delta betveen babai(t1) and babai(t2)
                eucl = [ (tmp@tmp)**0.5 for tmp in dbatch.T ]
                infnrm = [ np.max(np.abs(tmp)) for tmp in dbatch.T ]

                # - - - Admissibility: check that babai(guess2 + err) = babai(guess2) + babai(err) - - -

                true_err = np.concatenate([e,-s[:-self.kappa]])[-self.cd:]
                tmp0batch = sec_proj2_cols + true_err[:,None]
                W = proj_submatrix_modulus_blas( self.R[-self.cd:,-self.cd:], tmp0batch, dim=self.cd )
                tmp0batch = tmp0batch + self.R[-self.cd:,-self.cd:]@W

                tmp1batch = sec_proj2_cols
                W = proj_submatrix_modulus_blas( self.R[-self.cd:,-self.cd:], tmp1batch, dim=self.cd )
                tmp1batch = tmp1batch + self.R[-self.cd:,-self.cd:]@W

                true_err = np.atleast_2d(true_err).T.astype(np.float64)
                W = proj_submatrix_modulus_blas( self.R[-self.cd:,-self.cd:], true_err, dim=self.cd )
                tmp2 = true_err + self.R[-self.cd:,-self.cd:]@W

                is_adm = []
                tmp12 = tmp0batch - (tmp1batch+tmp2)
                for col in tmp12.T:
                    okay = all(np.isclose(col,0.0,atol=0.001))
                    is_adm.append(okay)

            return eucl, infnrm, sum(is_adm) 


        # For each (b,s,e) in bse, run n_trials of _trial_worker (parallelised)
        is_adm_num = 0
        is_adm_nums = []
        target_num = 0
        for b, s, e in self.bse[start:end]:
            mindd = float('inf')
            minddinf = float('inf')
            s_correct_guess = s[-self.kappa:]

            n_trials_normalized = (n_trials)//num_per_batch #num_per_batch used to be one
            if n_workers is None or n_workers <= 1:
                # sequential fallback
                for tries in range(n_trials_normalized):
                    if tries != 0 and tries % 10 == 0:
                        print(f"{tries} out of {n_trials_normalized} done")
                    eucl, infnrm, is_adm = _trial_worker(tries, num_per_batch, b, s_correct_guess,s,e)
                    is_adm_num+=is_adm
                    if np.min(eucl) < mindd:
                        mindd = np.min(eucl)
                    if np.min(infnrm) < minddinf:  #TODO: see if admissibility == minimizing the norm
                        minddinf = np.min(infnrm)
            else:
                # parallel execution using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = {
                        ex.submit(_trial_worker, tries, num_per_batch, b, s_correct_guess, s, e): tries
                        for tries in range(n_trials_normalized)
                    }

                    for fut in as_completed(futures):
                        tries = futures[fut]
                        if tries != 0 and tries % 10 == 0:
                            print(f"{tries} out of {n_trials_normalized} done")

                        try:
                            eucl, infnrm, is_adm = fut.result()
                        except Exception:
                            print(f"\n_trial_worker crashed for tries={tries}")
                            traceback.print_exc()          # prints full traceback to stderr
                            continue

                        is_adm_num += is_adm
                        mindd = min(mindd, float(np.min(eucl)))
                        minddinf = min(minddinf, float(np.min(infnrm)))
            target_num += 1
            print(f"mindd, minddinf: {mindd, minddinf}")
            print(f"is_adm_num: {is_adm_num} | {n_trials_normalized*num_per_batch}")
            if target_num%10==0:
                print(f"{target_num} out of {end-start} targets done", flush=True)
            # is_adm_nums.append(is_adm_num)
            is_adm_nums.append(is_adm_num)
            is_adm_num = 0
            mindds.append(mindd)
            minddinfs.append(minddinf)
        print(mindds)
        print(minddinfs)
        return minddinfs, mindds, is_adm_nums


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
      - runs check_correct_guess and check_pairs_guess_dist (correct/incorrect)
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
                G = GSO.Mat( lwe_instance.H, float_type="dd" )
                G.update_gso()

                succ, iters = lwe_instance.check_correct_guess()
                if succ/iters >= babai_succ_rate:
                    print(f"Stopped lattice reduction [{idx}] at {beta}: {succ/iters} >= {babai_succ_rate}")
                    break
                else:
                    print(f"Continuing reduction [{idx}]: {beta}: {succ/iters} < {babai_succ_rate}")
            

        # save reduced instance to disk so future runs can reuse it
        os.makedirs(in_path, exist_ok=True)
        lwe_instance.dump_on_disc(fullpath)
        print(f"[inst {idx}] dumped to {fullpath}")

    finally:
        lwe_instance = BatchAttackInstance.load_from_disc(fullpath)
        lwe_instance.cd = cd #technically, this should not be a field?
        loaded = True
        print(f"[inst {idx}] loaded from {fullpath}")


    # run experiments
    print(f"[inst {idx}] check_correct_guess()")
    succnum, itnum = lwe_instance.check_correct_guess()
    print(f"[inst {idx}] check_correct_guess -> ({succnum}, {itnum})")

    print(f"[inst {idx}] check_pairs_guess_dist(correct=False)")
    infdiff_incorrect, mindds_incorrect, is_adm_num_incorrect = lwe_instance.check_pairs_guess_dist(n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=False)

    print(f"[inst {idx}] check_pairs_guess_dist(correct=True)")
    infdiff_correct, mindds_correct, is_adm_num_correct       = lwe_instance.check_pairs_guess_dist(n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=True)
    
    print(f"correct adm: {is_adm_num_correct}")
    print(f"incorrect adm: {is_adm_num_incorrect}")

    return {
        "version": "0.0.1",  #file format version
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
        "infdiff_correct": infdiff_correct,
        "infdiff_incorrect": infdiff_incorrect,
        "mindds_correct": mindds_correct,
        "mindds_incorrect": mindds_incorrect,
        "loaded": loaded,
    }

def main():
    # outer parallelism: number of independent BatchAttackInstance runs
    n_workers = 1  # set this >1 to parallelize across instances
    n_lats = 1  # number of lattices    #5
    n_tars = 5 ## per-lattice instances #20
    n_trials = 1024*8 #8192*4          # per-lattice-instance trials in check_pairs_guess_dist. SHOULD be div'e by num_per_batch
    num_per_batch = 256 #8192
    inner_n_workers = 2   # threads for inner parallelism

    assert n_trials%num_per_batch == 0, f"n_trials should be divisible by num_per_batch"

    n, m, q = 100, 100, 4096
    seed_base = 0
    dist_s, dist_param_s, dist_e, dist_param_e = "ternary_sparse", 28, "binomial", 2
    kappa = 20
    cd = 50
    beta_max = 30
    babai_succ_rate = 0.97

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
    for r in results:
        if "error" in r:
            print(f"instance {r['idx']} ERROR: {r['error']}")
        else:
            print(f"instance {r['idx']} (seed {r['seed']}, file {r['filename']}, loaded={r['loaded']})")
            print(f"  check_correct_guess: ({r['succnum']}, {r['itnum']})")
            print(f"  infdiff_correct  (len {len(r['infdiff_correct'])}): {r['infdiff_correct']}")
            print(f"  infdiff_incorrect(len {len(r['infdiff_incorrect'])}): {r['infdiff_incorrect']}")

    return results


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
      - runs check_correct_guess and check_pairs_guess_dist (correct/incorrect)
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
                G = GSO.Mat( lwe_instance.H, float_type="dd" )
                G.update_gso()

                succ, iters = lwe_instance.check_correct_guess()
                if succ/iters >= babai_succ_rate:
                    print(f"Stopped lattice reduction [{idx}] at {beta}: {succ/iters} >= {babai_succ_rate}")
                    break
                else:
                    print(f"Continuing reduction [{idx}]: {beta}: {succ/iters} < {babai_succ_rate}")
            

        # save reduced instance to disk so future runs can reuse it
        os.makedirs(in_path, exist_ok=True)
        lwe_instance.dump_on_disc(fullpath)
        print(f"[inst {idx}] dumped to {fullpath}")

    finally:
        lwe_instance = BatchAttackInstance.load_from_disc(fullpath)
        lwe_instance.cd = cd #technically, this should not be a field?
        loaded = True
        print(f"[inst {idx}] loaded from {fullpath}")


    # run experiments
    print(f"[inst {idx}] check_correct_guess()")
    succnum, itnum = lwe_instance.check_correct_guess()
    print(f"[inst {idx}] check_correct_guess -> ({succnum}, {itnum})")

    print(f"[inst {idx}] check_pairs_guess_dist(correct=False)")
    infdiff_incorrect, mindds_incorrect, is_adm_num_incorrect = lwe_instance.check_pairs_guess_dist(n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=False)

    print(f"[inst {idx}] check_pairs_guess_dist(correct=True)")
    infdiff_correct, mindds_correct, is_adm_num_correct       = lwe_instance.check_pairs_guess_dist(n_trials=n_trials, n_workers=inner_n_workers, num_per_batch=num_per_batch, correct=True)
    
    print(f"correct adm: {is_adm_num_correct}")
    print(f"incorrect adm: {is_adm_num_incorrect}")

    return {
        "version": "0.0.1",  #file format version
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
        "infdiff_correct": infdiff_correct,
        "infdiff_incorrect": infdiff_incorrect,
        "mindds_correct": mindds_correct,
        "mindds_incorrect": mindds_incorrect,
        "loaded": loaded,
    }

def main():
    # outer parallelism: number of independent BatchAttackInstance runs
    n_workers = 1  # set this >1 to parallelize across instances
    n_lats = 1  # number of lattices    #5
    n_tars = 5 ## per-lattice instances #20
    n_trials = 1024*8 #8192*4          # per-lattice-instance trials in check_pairs_guess_dist. SHOULD be div'e by num_per_batch
    num_per_batch = 256 #8192
    inner_n_workers = 2   # threads for inner parallelism

    assert n_trials%num_per_batch == 0, f"n_trials should be divisible by num_per_batch"

    n, m, q = 100, 100, 4096
    seed_base = 0
    dist_s, dist_param_s, dist_e, dist_param_e = "ternary_sparse", 28, "binomial", 2
    kappa = 20
    cd = 50
    beta_max = 42
    babai_succ_rate = 0.97

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
    for r in results:
        if "error" in r:
            print(f"instance {r['idx']} ERROR: {r['error']}")
        else:
            print(f"instance {r['idx']} (seed {r['seed']}, file {r['filename']}, loaded={r['loaded']})")
            print(f"  check_correct_guess: ({r['succnum']}, {r['itnum']})")
            print(f"  infdiff_correct  (len {len(r['infdiff_correct'])}): {r['infdiff_correct']}")
            print(f"  infdiff_incorrect(len {len(r['infdiff_incorrect'])}): {r['infdiff_incorrect']}")

    return results

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    try:
        results = main()
        now = get_current_datetime()
        with open(f"res_{now}.pkl","wb") as file:
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
            print(f"  infdiff_correct  (len {len(r['infdiff_correct'])}): {r['infdiff_correct']}")
            print(f"  infdiff_incorrect(len {len(r['infdiff_incorrect'])}): {r['infdiff_incorrect']}")
        