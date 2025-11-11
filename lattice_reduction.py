import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, BKZ as BKZ_FPYLLL
from fpylll.util import gaussian_heuristic, ReductionError
from fpylll.algorithms.bkz2 import BKZReduction
from math import sqrt, ceil, floor, log, exp

import time, os
from random import randrange

from blaster import reduce
# def bkz_reduce(B, U, U_seysen, lll_size, delta, depth,
#                beta, bkz_tours, bkz_size, tprof, tracers, debug, use_seysen):
"""
Perform BLASter's BKZ reduction on basis B, and keep track of the transformation in U.
If `depth` is supplied, BLASter's deep-LLL is called in between calls of the SVP oracle.
Otherwise BLASter's LLL is run.
"""

try:
  from g6k import Siever, SieverParams
  from g6k.algorithms.bkz import pump_n_jump_bkz_tour
  from g6k.utils.stats import dummy_tracer
except ImportError:
  raise ImportError("g6k not installed")

def flatter_interface( fpylllB ):
    #import os
    basis = '[' + fpylllB.__str__() + ']'
    #print(basis)
    seed = randrange(2**32)
    filename = f"lat{seed}.txt"
    filename_out = f"redlat{seed}.txt"
    with open(filename, 'w') as file:
        file.write( "["+fpylllB.__str__()+"]" )
    
    out = os.system( "flatter " + filename + " > " + filename_out )
    time.sleep(float(0.05))
    os.remove( filename )
    
    B = IntegerMatrix.from_file( filename_out )
    os.remove( filename_out )
    return B

class LatticeReduction:
    def __init__(self,B):
        self.B = flatter_interface( IntegerMatrix.from_matrix( B ) )


    def __call__(self, lll_size  = 64, delta: float = 0.99, cores  = 1, debug  = False,
        verbose  = False, logfile  = None, anim  = None, depth  = 0,
        use_seysen  = False, beta=2, bkz_tours=1, bkz_size=64, **kwds):
        
        use_blaster=True
        if beta<40:
            G = GSO.Mat(self.B, float_type="dd",
            U=IntegerMatrix.identity(self.B.nrows, int_type=self.B.int_type),
            UinvT=IntegerMatrix.identity(self.B.nrows, int_type=self.B.int_type))
            G.update_gso()
            lll = LLL.Reduction(G)
            lll()

            bkz = BKZReduction( G )
            par = BKZ_FPYLLL.Param(
                beta,
                strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
                max_loops=bkz_tours,
                flags=BKZ_FPYLLL.MAX_LOOPS
            )
            bkz(par)

            self.B = bkz.M.B
        elif beta<65 and use_blaster:
            # print( np.array([list(b) for b in self.B]).shape[1] )
            self.B = reduce(np.array([list(tmp) for tmp in self.B]).transpose(), lll_size=lll_size, delta=delta, cores=cores, debug=debug,
            verbose=verbose, logfile=logfile, anim=None, depth=depth,
            use_seysen=use_seysen, bkz_tours=bkz_tours, beta=beta)[1].transpose()
            self.B=IntegerMatrix.from_matrix(self.B)
        else:
            G = GSO.Mat(self.B, float_type="dd", #WARNING: this will fail on ARM
            U=IntegerMatrix.identity(self.B.nrows, int_type=self.B.int_type),
            UinvT=IntegerMatrix.identity(self.B.nrows, int_type=self.B.int_type))
            G.update_gso()
            
            params_sieve = SieverParams()
            params_sieve['threads'] = cores

            g6kobj = Siever(G, params_sieve)
            

            for t in range(bkz_tours): #pnj-bkz is oblivious to ntours
                    print(f"@beta={beta} tour:{t}")
                    pump_n_jump_bkz_tour(g6kobj, dummy_tracer, beta)
            self.B = G.B

            """
            G = GSO.Mat(self.B, float_type="mpfr",
            U=IntegerMatrix.identity(self.B.nrows, int_type=self.B.int_type),
            UinvT=IntegerMatrix.identity(self.B.nrows, int_type=self.B.int_type))
            G.update_gso()
            lll = LLL.Reduction(G)
            lll()

            bkz = BKZReduction( G )
            par = BKZ_FPYLLL.Param(
                beta,
                strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
                max_loops=bkz_tours,
                flags=BKZ_FPYLLL.MAX_LOOPS
            )
            bkz(par)

            self.B = bkz.M.B
            """
        return self.B
