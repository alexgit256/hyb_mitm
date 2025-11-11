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

from utils import *

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

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
            print(f"lol: {(s@self.A+e-b)%self.q}")
            s_correct_guess = s[-self.kappa:]
            sec_proj = ( s_correct_guess @ self.C )
            t = np.concatenate( [b,(self.n-self.kappa)*[0]] )
            t -= sec_proj

            print(t.shape)
            bab = G.babai(t)

            tshift = t - G.B.multiply_left(bab)
            
            diff = tshift-np.concatenate([e,-s[:-self.kappa]])
            if all( np.isclose(diff, 0.0, atol=1e-10) ):
                succnum+=1
            print(f"diff: {diff}")
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
                # print(f"msk: {msk}")
                sguess_1 = s_correct_guess*msk
                sguess_2 = sguess_1 - s_correct_guess
                # print(f"s: {s_correct_guess-(sguess_1-sguess_2)}")
                # print(f"00: {s_correct_guess@self.C-np.concatenate( [b,(self.m-self.kappa)*[0]] )}")

                sec_proj1 = ( sguess_1 @ self.C )
                t1 = np.concatenate( [b,(self.n-self.kappa)*[0]] )
                t1 -= sec_proj1
                # print(t.shape)
                bab = G.babai(t1)
                tshift1 = t1 - G.B.multiply_left(bab)

                sec_proj2 = ( sguess_2 @ self.C )
                t2 = -sec_proj2
                # print(t.shape)
                bab = G.babai(t2)
                tshift2 = t2 - G.B.multiply_left(bab) #should be close to tshift1 with some proba by assumption

                d = tshift1-tshift2
                # print(f"close? {(d@d)**0.5, np.max(np.abs(d))}")
                infnrm = np.max(np.abs(d))
                if (d@d)**0.5 < mindd:
                    mindd = (d@d)**0.5
                if infnrm<minddinf:
                    minddinf = infnrm
                if infnrm < 4:
                    print("- - - A - - -")
                    print(msk)

                # t_correct_guess = (s_correct_guess @ self.C )
                # t_correct_guess = np.concatenate( [b,(self.m-self.kappa)*[0]] ) - t_correct_guess
                tmp = G.B.multiply_left( G.babai(t1-t2) )
                tmp = (t1-t2)-tmp
                # print(((tmp)))
                # print((tshift1-tshift2))
            print(f"mindd, minddinf: {mindd, minddinf}")
            mindds.append(mindd)
            minddinfs.append(minddinf)
        print(mindds)
        print(minddinfs)
        print(f"gh: {gaussian_heuristic(G.r())**0.5}")

    """
    def check_correct_pairs_guess_MM(self, start=0, end=None, n_trials=10, n_workers=1):
        Constructs correct guesses s=s1-s2 for the MitM attack on LWE and checks if Babai(t+s2) == Babai(s1)
        n_trials: number of decompositions of s.
        n_workers: number of parralel checks.
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
                
                tshift1 = proj_submatrix_modulus(G,t1,dim=self.cd)

                sec_proj2 = ( sguess_2 @ self.C )
                t2 = sec_proj2
                
                tshift2 = proj_submatrix_modulus(G,t2,dim=self.cd)

                d = tshift1-tshift2
                d =  np.array( G.from_canonical(d)[-self.cd:] )

                print(f"close? {(d@d)**0.5, np.max(np.abs(d))} | {(tshift1@tshift1)**0.5, (tshift2@tshift2)**0.5}")
                infnrm = np.max(np.abs(d))
                if (d@d)**0.5 < mindd:
                    mindd = (d@d)**0.5
                if infnrm<minddinf:
                    minddinf = infnrm
                if infnrm < 4:
                    print("- - - A - - -")
                    print(msk)

            print(f"mindd, minddinf: {mindd, minddinf}")
            mindds.append(mindd)
            minddinfs.append(minddinf)
        print(mindds)
        print(minddinfs)
        print(f"gh: {gaussian_heuristic(G.r())**0.5}")
    """
                
    def check_correct_pairs_guess_MM(self, start=0, end=None, n_trials=10, n_workers=1):
        """
        Constructs correct guesses s=s1-s2 for the MitM attack on LWE and checks if Babai(t+s2) == Babai(s1)
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
        def _trial_worker(trie_idx, b, s_correct_guess):
            # create a local RNG to avoid shared-state race
            seed = (os.getpid() ^ trie_idx ^ int(time.time_ns()))
            rng = random.Random(seed)

            # replicate original mask length logic (original code produced 0 or 1 because expression reduces to 2)
            msk_sublen = rng.randrange(self.kappa//2 - self.kappa//2 + 2)
            msk = msk_sublen*[1] + (self.kappa - msk_sublen)*[0]
            rng.shuffle(msk)

            # ensure numpy arrays for elementwise ops (keeps behaviour stable regardless of input type)
            s_corr_np = np.asarray(s_correct_guess)
            msk_np = np.asarray(msk)
            sguess_1 = s_corr_np * msk_np
            sguess_2 = sguess_1 - s_corr_np

            # compute projections and shifts (exact same calls as original)
            sec_proj1 = ( sguess_1 @ self.C )
            t1 = np.concatenate( [b,(self.n-self.kappa)*[0]] )
            t1 -= sec_proj1

            tshift1 = proj_submatrix_modulus(G,t1,dim=self.cd)

            sec_proj2 = ( sguess_2 @ self.C )
            t2 = sec_proj2

            tshift2 = proj_submatrix_modulus(G,t2,dim=self.cd)

            d = tshift1 - tshift2
            d = np.array( G.from_canonical(d)[-self.cd:] )

            eucl = (d @ d)**0.5
            infnrm = np.max(np.abs(d))

            # return the distances so caller can reduce minima
            return eucl, infnrm

        # For each (b,s,e) in bse, run n_trials of _trial_worker (parallelised)
        for b, s, e in self.bse[start:end]:
            mindd = self.q
            minddinf = self.q
            s_correct_guess = s[-self.kappa:]

            if n_workers is None or n_workers <= 1:
                # sequential fallback (keeps original print behaviour)
                for tries in range(n_trials):
                    if tries != 0 and tries % 1000 == 0:
                        print(f"{tries} out of {n_trials} done")
                    eucl, infnrm = _trial_worker(tries, b, s_correct_guess)

                    if eucl < mindd:
                        mindd = eucl
                    if infnrm < minddinf:
                        minddinf = infnrm
                    if infnrm < 4:
                        print("- - - A - - -")
            else:
                # parallel execution using ThreadPoolExecutor
                # use threads to avoid pickling heavy objects (G, mpfr, etc.)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    # submit all trials
                    futures = { ex.submit(_trial_worker, tries, b, s_correct_guess): tries for tries in range(n_trials) }

                    # as each completes, update minima (and optionally print progress)
                    completed = 0
                    for fut in as_completed(futures):
                        tries = futures[fut]
                        completed += 1
                        if tries != 0 and tries % 1000 == 0:
                            print(f"{tries} out of {n_trials} done")
                        try:
                            eucl, infnrm = fut.result()
                        except Exception as exc:
                            # keep robust: print and continue
                            print(f"trial {tries} raised {exc!r}")
                            continue

                        if eucl < mindd:
                            mindd = eucl
                        if infnrm < minddinf:
                            minddinf = infnrm
                        if infnrm < 4:
                            print("- - - A - - -")
                            print(f"(trial {tries} mask triggered small inf norm)")

            print(f"mindd, minddinf: {mindd, minddinf}")
            mindds.append(mindd)
            minddinfs.append(minddinf)
        print(mindds)
        print(minddinfs)
        return minddinfs


if __name__ == "__main__":
    n, m, q, n_tars = 120, 120, 3329, 20
    dist_s,dist_param_s,dist_e,dist_param_e = "ternary_sparse", 12, "binomial", 2
    kappa = 20
    cd = 32
    lwe_instance = BatchAttackInstance(
        n=n, m=m, q=q, n_tars=n_tars,
        dist_s=dist_s,dist_param_s=dist_param_s,dist_e=dist_e,dist_param_e=dist_param_e,
        kappa=kappa, cd=cd
    )

    for beta in list(range(30,31))+list(range(40,53,1)):
        t0 = perf_counter()
        lwe_instance.reduce(beta=beta,bkz_tours=2,cores=6,depth=4)
        print(f"bkz-{beta} done in {perf_counter()-t0}")
    print("chk0")
    print( lwe_instance.check_correct_guess() )
    lwe_instance.check_correct_pairs_guess_MM(n_trials=20, n_workers=5)
            
        