from lwe_gen import generateLWEInstances
from lattice_reduction import LatticeReduction
import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, FPLLL
from fpylll.util import gaussian_heuristic
FPLLL.set_precision(80)

from time import perf_counter
from random import randrange, shuffle

class BatchAttackInstance:
    def __init__(self, **kwargs):
        self.A = None
        self.q = None
        self.bse = None

        self.LR = None
        if not kwargs.get("lwe_instance") is  None:
            self.A, self.q, self.bse = kwargs.get("lwe_instance") 
            self.n = len(self.bse[0]["s"])
            self.m = len(self.bse[0]["e"])
            self.n_tars = len(self.bse)
        else:
            self.n = kwargs.get("n") 
            self.m = kwargs.get("m") 
            self.q = kwargs.get("q")
            self.n_tars = kwargs.get("n_tars")  
            self.dist_s,self.dist_param_s,self.dist_e,self.dist_param_e = kwargs.get("dist_s"), kwargs.get("dist_param_s"), kwargs.get("dist_e") , kwargs.get("dist_param_e")
            self.n_tars = kwargs.get("n_tars")

            self.A, self.q, self.bse  = generateLWEInstances(self.n,self.q,self.dist_s,self.dist_param_s,self.dist_e,self.dist_param_e,ntar=self.n_tars)

        self.kappa = kwargs.get("kappa") #guessing dim
        self.cd = kwargs.get("cd") #CVP dim

        assert self.m+self.n-(self.kappa+self.cd) >=0, f"Too large attack dimensions!"

        n = self.n
        m = self.m
        B = [ [int(0) for i in range(2*m)] for j in range(2*n) ]
        for i in range( n ):
            B[i][i] = int( q )
        for i in range(n, 2*n):
            B[i][i] = 1
        for i in range(n, 2*n):
            for j in range(m):
                B[i][j] = int( self.A[i-n,j] )

        self.H = B[:len(B)-self.kappa] #the part of basis to be reduced
        self.H = IntegerMatrix.from_matrix( [ h11[:len(B)-self.kappa] for h11 in self.H  ] )

        self.C = [ b[:-self.kappa] for b in B[len(B)-self.kappa:] ] #the guessing matrix 


    def reduce(self,beta, bkz_tours, bkz_size=64, lll_size = 64, delta = 0.99, cores=1, debug= False,
        verbose= True, logfile= None, anim= None, depth = 4,
        use_seysen = False, **kwds):
        if self.LR is None:
            self.LR = LatticeReduction( self.H )

        self.H = self.LR(lll_size=lll_size, delta=delta, cores=cores, debug=debug,
            verbose=verbose, logfile=logfile, anim=anim, depth=depth,
            use_seysen=use_seysen, beta=beta, bkz_tours=bkz_tours, bkz_size=64)
        
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
            t = np.concatenate( [b,(self.m-self.kappa)*[0]] )
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
                t1 = np.concatenate( [b,(self.m-self.kappa)*[0]] )
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
                



if __name__ == "__main__":
    n, m, q, n_tars = 144, 144, 3329, 10
    dist_s,dist_param_s,dist_e,dist_param_e = "ternary_sparse", 70, "binomial", 2
    kappa = 22
    cd = 50
    lwe_instance = BatchAttackInstance(
        n=n, m=m, q=q, n_tars=n_tars,
        dist_s=dist_s,dist_param_s=dist_param_s,dist_e=dist_e,dist_param_e=dist_param_e,
        kappa=kappa, cd=cd
    )

    for beta in list(range(30,31))+list(range(40,66,5)):
        t0 = perf_counter()
        lwe_instance.reduce(beta=beta,bkz_tours=2,cores=6,depth=4)
        print(f"bkz-{beta} done in {perf_counter()-t0}")
    print("chk0")
    print( lwe_instance.check_correct_guess() )
    lwe_instance.check_correct_pairs_guess(n_trials=600)
            
        