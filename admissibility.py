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

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import concurrent.futures

from datetime import datetime

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
            C = [ b[:len(B)-self.kappa] for b in B[len(B)-self.kappa:] ] #the guessing matrix 
        self.H, self.C = H, C


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
        
    def check_correct_guess(self, start=0, end=None, n_trials=1):
        if end is None:
            end = len(self.bse)
        G = GSO.Mat( IntegerMatrix.from_matrix( self.H ), float_type="dd" )
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
    
    def check_correct_guess_w_babai(self, start=0, end=None, n_trials=1):
        if end is None:
            end = len(self.bse)
        G = GSO.Mat( IntegerMatrix.from_matrix( self.H ), float_type="dd" )
        G.update_gso()
        succnum=0
        itnum=0
        for b, s, e in self.bse[start:end]:
            itnum+=1
            s_correct_guess = s[-self.kappa:]
            sec_proj = ( s_correct_guess @ self.C )
            t = np.concatenate( [b,(self.n-self.kappa)*[0]] )
            t -= sec_proj

            # shift2 = proj_submatrix_modulus(G,t,dim=G.d)
            # tshift = shift2

            shift2, c = proj_submatrix_modulus(G,t,dim=self.cd,coords_too=True)
            tst = proj_submatrix_modulus(G,t-shift2,dim=self.cd)
            assert all(np.isclose(tst,0.0,atol=0.001)), f"Nooo! {tst}"
            
            bab = G.babai(t-shift2)
            # print(f"bab: {bab[-self.cd:]}")
            # print(f"c: {c}")

            tshift = t - G.B.multiply_left(bab)
            
            diff = tshift-np.concatenate([e,-s[:-self.kappa]])
            if all( np.isclose(diff, 0.0, atol=1e-10) ):
                succnum+=1
            else:
                pass
                # print(f"No: {diff}")
        return succnum, itnum
    
    def check_correct_pairs_guess(self, start=0, end=None, n_trials=10):
        if end is None:
            end = len(self.bse)
        G = GSO.Mat( IntegerMatrix.from_matrix( self.H ), float_type="mpfr" )
        G.update_gso()
        succnum=0
        itnum=0
        mindds, minddinfs = [], []
        for b, s, e in self.bse[start:end]:
            mindd = self.q
            minddinf = self.q
            s_correct_guess = s[-self.kappa:]
            for tries in range(n_trials):
                if tries!=0 and tries%1000==0:
                    print(f"{tries} out of {n_trials} done")
                itnum+=1
                msk_sublen = randrange(self.kappa//2-self.kappa//2+2)
                msk = msk_sublen*[1] + (self.kappa - msk_sublen)*[0]
                shuffle(msk)

                sguess_1 = s_correct_guess*msk
                sguess_2 = sguess_1 - s_correct_guess

                sec_proj1 = ( sguess_1 @ self.C )
                t1 = np.concatenate( [b,(self.n-self.kappa)*[0]] )
                t1 -= sec_proj1

                bab = G.babai(t1)
                tshift1 = t1 - G.B.multiply_left(bab)

                sec_proj2 = ( sguess_2 @ self.C )
                t2 = -sec_proj2

                bab = G.babai(t2)
                tshift2 = t2 - G.B.multiply_left(bab) #should be close to tshift1 with some proba by assumption

                d = tshift1-tshift2
                d = G.from_canonical( d )
                d = (self.H.nrows-self.cd)*[0] + list(d[-self.cd:])
                d = np.asarray( G.to_canonical(d) )
                infnrm = np.max(np.abs(d))
                if (d@d)**0.5 < mindd:
                    mindd = (d@d)**0.5
                if infnrm<minddinf:
                    minddinf = infnrm
                if infnrm < 4:
                    print("- - - A - - -")
                    print(msk)

                tmp = G.B.multiply_left( G.babai(t1-t2) )
                tmp = (t1-t2)-tmp
            # print(f"mindd, minddinf: {mindd, minddinf}")
            mindds.append(mindd)
            minddinfs.append(minddinf)
        print(mindds)
        print(minddinfs)
        print(f"gh: {gaussian_heuristic(G.r())**0.5}")

                
    def check_pairs_guess_MM(self, start=0, end=None, correct=True, n_trials=10, n_workers=1):
        """
        Constructs correct (if correct==True) guesses s=s1-s2 for the MitM attack on LWE and checks if Babai(t+s2) == Babai(s1)
        n_trials: number of decompositions of s.
        n_workers: number of parallel checks (threads).
        Returns:
            minddinfs : list of per-(b,s,e) minimal infinity norms found across trials
        """
        if end is None:
            end = len(self.bse)
        G = GSO.Mat( IntegerMatrix.from_matrix( self.H ), float_type="mpfr" )
        G.update_gso()

        mindds, minddinfs = [], []

        # Helper that performs a single trial for current (b, s_correct_guess)
        def _trial_worker(trie_idx, b, s_correct_guess,s,e):
            # create a local RNG to avoid shared-state race
            seed = (os.getpid() ^ trie_idx ^ int(time.time_ns()))
            rng = random.Random(seed)

            msk_sublen = rng.randrange(self.kappa//2, self.kappa)
            msk = msk_sublen*[1] + (self.kappa - msk_sublen)*[0]
            # msk = (self.kappa)*[1]
            rng.shuffle(msk)

            # ensure numpy arrays for elementwise ops
            s_corr_np = np.asarray(s_correct_guess)
            msk_np = np.asarray(msk)

            if correct:
                # s_delta = np.asarray( [ (randrange(-1,2)) for j in range(self.kappa) ] )
                # sguess_1 = s_delta
                # sguess_2 = sguess_1 - s_corr_np
                sguess_1 = s_corr_np * msk_np
                sguess_2 = sguess_1 - s_corr_np
            else:
                sguess_1 = np.asarray([ randrange(-1,2) for _ in range(len(s_corr_np)) ])
                sguess_2 = np.asarray([ randrange(-1,2) for _ in range(len(s_corr_np)) ])

            # compute projections and shifts
            sec_proj1 = ( sguess_1 @ self.C )
            t = np.concatenate( [b,(self.n-self.kappa)*[0]] )
            tshift1 = proj_submatrix_modulus(G,t-sec_proj1,dim=self.cd)

            # babt = proj_submatrix_modulus(G,t1-sec_proj1,dim=self.cd) #error coming from t
            # babsec_proj1 = proj_submatrix_modulus(G,sec_proj1,dim=self.cd) #error coming from A2
            # true_err = proj_submatrix_modulus(G, np.concatenate([e,-s[:-self.kappa]]), dim=self.cd ) #true error
            # if all( np.isclose((true_err - (babt + babsec_proj1)), 0.0, atol=1e-7) ):
            #     print(f"Admissible pair!")
            # else: 
            #     pass
                

            sec_proj2 = ( sguess_2 @ self.C )
            t2 = sec_proj2
            tshift2 = proj_submatrix_modulus(G,sec_proj2,dim=self.cd)

            d = tshift1 - tshift2  #NP(t-sec_proj1) + NP(sec_proj2) =conj= NP(t) - ( NP(sec_proj1)-NP(sec_proj2) )
            d = G.from_canonical( d )
            d = (self.H.nrows-self.cd)*[0] + list(d[-self.cd:])
            d = np.asarray( G.to_canonical(d) )

            true_err = np.concatenate([e,-s[:-self.kappa]])
            true_err_gh = proj_submatrix_modulus(G,true_err,dim=self.cd)
            npsp1 = proj_submatrix_modulus(G,sec_proj1,dim=self.cd)
            lol = proj_submatrix_modulus(G,npsp1-true_err_gh,dim=self.cd)
            lol2 = ( (lol) - npsp1 + true_err_gh )
            is_adm = False
            if all(np.isclose(proj_submatrix_modulus(G,lol2,dim=self.cd),0.0,atol=0.001)):
                is_adm=True
            assert all(np.isclose(proj_submatrix_modulus(G,lol2,dim=self.cd),0.0,atol=0.001)), f"Babai is wrong"

            eucl = (d @ d)**0.5
            infnrm = np.max(np.abs(d))

            return eucl, infnrm, is_adm

        # For each (b,s,e) in bse, run n_trials of _trial_worker (parallelised)
        is_adm_num = 0
        for b, s, e in self.bse[start:end]:
            mindd = float('inf')
            minddinf = float('inf')
            s_correct_guess = s[-self.kappa:]

            if n_workers is None or n_workers <= 1:
                # sequential fallback
                for tries in range(n_trials):
                    if tries != 0 and tries % 1000 == 0:
                        print(f"{tries} out of {n_trials} done")
                    eucl, infnrm, is_adm = _trial_worker(tries, b, s_correct_guess,s,e)
                    is_adm_num+=is_adm
                    if eucl < mindd:
                        mindd = eucl
                    if infnrm < minddinf:
                        minddinf = infnrm
            else:
                # parallel execution using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = { ex.submit(_trial_worker, tries, b, s_correct_guess, s,e): tries for tries in range(n_trials) }

                    for fut in as_completed(futures):
                        tries = futures[fut]
                        if tries != 0 and tries % 1000 == 0:
                            print(f"{tries} out of {n_trials} done")
                        try:
                            eucl, infnrm, is_adm= fut.result()
                        except Exception as exc:
                            print(f"trial {tries} raised {exc!r}")
                            continue
                        is_adm_num+=is_adm
                        if eucl < mindd:
                            mindd = eucl
                        if infnrm < minddinf:
                            minddinf = infnrm

            print(f"mindd, minddinf: {mindd, minddinf}")
            mindds.append(mindd)
            minddinfs.append(minddinf)
        print(mindds)
        print(minddinfs)
        return minddinfs, mindds, is_adm_num


# NEW: worker to run one complete BatchAttackInstance experiment in a separate process
def run_single_instance(idx: int,
                        n: int, m: int, q: int, n_tars: int,
                        dist_s: str, dist_param_s: int,
                        dist_e: str, dist_param_e: int,
                        kappa: int, cd: int,
                        beta_max: int,
                        seed_base: int,
                        n_trials: int,
                        ntar: int,
                        inner_n_workers: int) -> Dict[str, Any]:
    """
    One worker:
      - seeds RNG
      - loads or creates + reduces a BatchAttackInstance
      - runs check_correct_guess and check_pairs_guess_MM (correct/incorrect)
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

        # same beta schedule as your original code: 30, then 40..beta_max-1
        for beta in list(range(30, 31)) + list(range(40, beta_max, 1)):
            t0 = perf_counter()
            lwe_instance.reduce(beta=beta, bkz_tours=2, 
                                cores=6, depth=4,
                                start=0, end=None)
            print(f"[inst {idx}] bkz-{beta} done in {perf_counter()-t0:.2f}s")

        # save reduced instance to disk so future runs can reuse it
        os.makedirs(in_path, exist_ok=True)
        lwe_instance.dump_on_disc(fullpath)
        print(f"[inst {idx}] dumped to {fullpath}")

    # run experiments
    print(f"[inst {idx}] check_correct_guess()")
    succnum, itnum = lwe_instance.check_correct_guess()
    print(f"[inst {idx}] check_correct_guess -> ({succnum}, {itnum})")

    
    print(f"[inst {idx}] check_correct_guess_w_babai()")
    succnum, itnum = lwe_instance.check_correct_guess_w_babai()
    print(f"[inst {idx}] check_correct_guess_w_babai -> ({succnum}, {itnum})")

    print(f"[inst {idx}] check_pairs_guess_MM(correct=True)")
    infdiff_correct, mindds_correct, is_adm_num = lwe_instance.check_pairs_guess_MM(n_trials=n_trials, n_workers=inner_n_workers, correct=True)

    print(f"[inst {idx}] check_pairs_guess_MM(correct=False)")
    infdiff_incorrect, mindds_incorrect = lwe_instance.check_pairs_guess_MM(correct=False, n_trials=n_trials, n_workers=inner_n_workers)

    return {
        "idx": idx,
        "seed": seed,
        'n': n,
        'm': m,
        'q': q,
        'beta_max': beta_max,
        'n_trials': n_trials,
        "filename": filename,
        "is_adm_num": is_adm_num,
        "succnum": succnum,
        "itnum": itnum,
        "infdiff_correct": infdiff_correct,
        "infdiff_incorrect": infdiff_incorrect,
        "mindds_correct": infdiff_correct,
        "mindds_incorrect": infdiff_incorrect,
        "loaded": loaded,
    }

def main():
    # outer parallelism: number of independent BatchAttackInstance runs
    max_workers = 2  # set this >1 to parallelize across instances
    n_lats = 2  # number of lattices    #5
    n_tars = 10 ## per-lattice instances #20
    n_trials = 1000          # per-lattice-instance trials in check_pairs_guess_MM
    inner_n_workers = 5    # threads for inner parallelism

    n, m, q = 128, 128, 3329
    seed_base = 0
    dist_s, dist_param_s, dist_e, dist_param_e = "ternary_sparse", 64, "binomial", 2
    kappa = 30
    cd = 45
    beta_max = 48

    os.makedirs(in_path, exist_ok=True)

    results = []
    # use processes for independent instances (CPU-bound)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
                n_tars,
                inner_n_workers
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
        