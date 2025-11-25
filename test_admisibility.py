from random import shuffle, randrange

import os
from fpylll import *
from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
from itertools import chain
import time
from time import perf_counter

from random import uniform

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np

from lattice_reduction import LatticeReduction
import pickle

from utils import uniform_in_ball

# your helper functions stay the same
def flatter_interface(fpylllB):
    basis = '[' + fpylllB.__str__() + ']'
    seed = randrange(2**32)
    filename = f"lat{seed}.txt"
    filename_out = f"redlat{seed}.txt"
    with open(filename, 'w') as file:
        file.write("[" + fpylllB.__str__() + "]")
    out = os.system("flatter " + filename + " > " + filename_out)
    time.sleep(0.05)
    os.remove(filename)

    B = IntegerMatrix.from_file(filename_out)
    os.remove(filename_out)
    return B

def babai(G, t, mod_red=False):
    c = G.babai(t)
    v = np.asarray(G.B.multiply_left(c))
    if mod_red:
        return np.asarray(t) - np.asarray(v)
    return np.asarray(v)

# global parameters
n = 80
num_tests = 250000
num_lats = 10
max_outer_workers = 5   # processes
max_inner_workers = 5   # threads *within* each process

gammas = np.linspace(0.3, 0.5, 9)

def process_single_lattice(lat_idx: int):
    """
    Do everything that used to be inside:
        for lat_idx in range(num_lats):
            ...
    and return whatever you care about (e.g. succs, succbabs).
    This function runs in a separate process.
    """

    # ---- lattice generation & BKZ ----

    if os.path.isfile( f"./lattices/lat{n}_{lat_idx}.pkl" ):
        with open(f"./lattices/lat{n}_{lat_idx}.pkl","rb") as file:
            B = pickle.load( file )
        G = GSO.Mat(B, float_type="double")
        G.update_gso()
    else:
        B = IntegerMatrix(n, n)
        B.randomize("qary", k=n//2, bits=14.2)

        G = GSO.Mat(B, float_type="double")
        G.update_gso()

        lll = LLL.Reduction(G)
        lll()

        LR = LatticeReduction(lll.M.B)

        for beta in list(range(40, n)):    # BKZ reduce the basis
            t0 = perf_counter()
            LR(beta=beta,
            bkz_tours=2,
            cores=max_inner_workers,
            depth=4,
            start=0,
            end=None)
            print(f"[lat {lat_idx}] BKZ-{beta} done in {perf_counter()-t0}")
        os.makedirs("lattices",exist_ok=True)
        
        with open(f"./lattices/lat{n}_{lat_idx}.pkl", "wb") as file:
            pickle.dump(LR.B, file)

    gh = gaussian_heuristic(G.r())**0.5   # if you prefer that

    # local worker for a single w (runs in threads within this *process*)
    def _worker(w):
        a = [uniform(-0.5, 0.5) for _ in range(n)]
        a_noise = np.asarray(G.to_canonical(a))

        tmp = G.babai(a_noise + w)
        succ_inc = 1 if all( np.isclose(tmp, 0.0, atol=1e-10) ) else 0
        succbab_inc = 1 if all( np.isclose(G.babai(w), 0.0, atol=1e-10) ) else 0
        return succ_inc, succbab_inc

    succs, succbabs = {}, {}

    for gamma in gammas:
        print(f"[lat {lat_idx}] gamma: {gamma}", flush=True)
        W = uniform_in_ball(num_tests, n, radius=gamma * gh)
        succ = 0
        succbab = 0

        # inner parallel loop (per gamma) – threads
        with ThreadPoolExecutor(max_workers=max_inner_workers) as ex:
            for it, (s_inc, sb_inc) in enumerate(ex.map(_worker, W), start=1):
                succ += s_inc
                succbab += sb_inc

                if it % 2000 == 0:
                    print(f"[lat {lat_idx}] it: {it} | {(succ, succbab)}", end=",")

        succs[gamma] = float(succ / num_tests)
        succbabs[gamma] = float(succbab / num_tests)

    return lat_idx, succs, succbabs

def process_single_box(lat_idx: int):
    if os.path.isfile( f"./lattices/lat{n}_{lat_idx}.pkl" ):
        with open(f"./lattices/lat{n}_{lat_idx}.pkl","rb") as file:
            B = pickle.load( file )
        G = GSO.Mat(B, float_type="double")
        G.update_gso()
    else:
        B = IntegerMatrix(n, n)
        B.randomize("qary", k=n//2, bits=14.2)

        G = GSO.Mat(B, float_type="double")
        G.update_gso()

        lll = LLL.Reduction(G)
        lll()

        LR = LatticeReduction(lll.M.B)

        for beta in list(range(40, n)):    # BKZ reduce the basis
            t0 = perf_counter()
            LR(beta=beta,
            bkz_tours=2,
            cores=max_inner_workers,
            depth=4,
            start=0,
            end=None)
            print(f"[lat {lat_idx}] BKZ-{beta} done in {perf_counter()-t0}")
        os.makedirs("lattices",exist_ok=True)
        
        with open(f"./lattices/lat{n}_{lat_idx}.pkl", "wb") as file:
            pickle.dump(LR.B, file)

    gh = gaussian_heuristic(G.r())**0.5
    def _worker(w):
        # local computation for a single w
        a = np.array( [uniform(-0.5, 0.5) for _ in range(n)] )
        ww = np.asarray( G.from_canonical( w ) )
        
        succ_inc = 1 if all( np.abs(a+ww) <= 0.500001 ) else 0
        succbab_inc = 1 if all( np.abs(ww) <= 0.500001 ) else 0
        return succ_inc, succbab_inc

    succs, succbabs = {}, {} 
    for gamma in gammas:
        print(f"gamma: {gamma}")
        W = uniform_in_ball(num_tests,n,radius=gamma*gh)
        succ = 0
        succbab = 0
        
        # tune max_workers to your CPU / GIL behavior
        max_workers = 12  # None -> default = number of processors * 5
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for it, (s_inc, sb_inc) in enumerate(ex.map(_worker, W), start=1):
                succ += s_inc
                succbab += sb_inc
        
                if it % 2000 == 0:
                    print(f"it: {it} | {(succ, succbab)}", end=",")
        
        # succ and succbab now contain the final counts
        
        succs[gamma], succbabs[gamma] = float(succ / num_tests), float(succbab / num_tests)
    return lat_idx, succs, succbabs

if __name__ == "__main__":
    # outer parallel loop – processes
    results = {}

    try:
        with ProcessPoolExecutor(max_workers=max_outer_workers) as outer_pool:
            futures = {
                outer_pool.submit(process_single_lattice, lat_idx): lat_idx
                for lat_idx in range(num_lats)
            }

            for fut in as_completed(futures):
                lat_idx, succs, succbabs = fut.result()
                results[lat_idx] = (succs, succbabs)
                print(f"\n[lattice {lat_idx}] finished.")
    except KeyboardInterrupt:
        # optional: nicer Ctrl-C behaviour
        print("\nInterrupted by user, shutting down workers.")
        # `cancel_futures=True` is Python 3.9+
        outer_pool.shutdown(wait=False, cancel_futures=True)
        raise

    # `results` now holds succs/succbabs dictionaries for each lattice index
    # do whatever post-processing you want here

    with open(f"exp_{n}.pkl", "wb") as file:
        pickle.dump(
            {"succbabs":succbabs,
             "succs": succs}
            ,file)
        
    print(f"- - - Now the sim - - -")
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=max_outer_workers) as outer_pool:
            futures = {
                outer_pool.submit(process_single_box, lat_idx): lat_idx
                for lat_idx in range(num_lats)
            }

            for fut in as_completed(futures):
                lat_idx, succs, succbabs = fut.result()
                results[lat_idx] = (succs, succbabs)
                print(f"\n[lattice {lat_idx}] sim finished.")
    except KeyboardInterrupt:
        # optional: nicer Ctrl-C behaviour
        print("\nInterrupted by user, shutting down workers.")
        # `cancel_futures=True` is Python 3.9+
        outer_pool.shutdown(wait=False, cancel_futures=True)
        raise

    # `results` now holds succs/succbabs dictionaries for each lattice index
    # do whatever post-processing you want here

    with open(f"exp_{n}_sim.pkl", "wb") as file:
        pickle.dump(
            {"succbabs":succbabs,
             "succs": succs}
            ,file)
    print()
    print(results)