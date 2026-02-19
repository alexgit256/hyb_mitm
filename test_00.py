import numpy as np

from functools import partial
from sys import stderr
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter

# Local imports
from blaster_core import \
    set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, ZZ_right_matmul
from size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce, nearest_plane
from stats import get_profile, rhf, slope, potential
from lattice_io import read_qary_lattice, write_lattice

from random import randrange, uniform

from copy import copy, deepcopy

from fpylll import *
from lattice_reduction import LatticeReduction

from utils import proj_submatrix_modulus_blas

def proj_submatrix_modulus_blas(R,T,dim=None):
    """

    Given Q, R, T, U where R is the R-factor, finds the coordinates of the corresponding babai-close lattice vectors.
    :param R: upper-triangular basis of a lattice.
    :param T: matrix containing many targets requiring reduction.
    :param U: the output transformation used to reduce T wrt R.
    """
    m, n = np.shape(T) #m - num basis vects, 

    d = np.shape(R)[1]
    if dim is None:
        dim = d
    elif not (1 <= dim <= d):
        raise ValueError("dim must be in [1, G.B.nrows]")

    # Tproj = T[d-dim:] #projection of arbitrary many columns of T onto the last dim coords
    # Rproj = R[d-dim:,d-dim:] #R-factor of the last dim dimensional projective sublattice
    U = np.zeros( (dim,n),dtype=np.int64 )

    nearest_plane(R,T,U)

    return U

# def nearest_plane(R, T, U):
"""
Perform Babai's Nearest Plane algorithm on multiple targets (all the columns of T), with
respect to the upper-triangular basis R.
This function updates T <- T + RU such that `T + RU` is in the fundamental Babai domain.
Namely, |(T + RU)_{ij}| <= 0.5 R_ii.

Complexity: O(N n^{omega-1}) if R is a `n x n` matrix, T is a `n x N` matrix, and `N >= n`.

:param R: upper-triangular basis of a lattice.
:param T: matrix containing many targets requiring reduction.
:param U: the output transformation used to reduce T wrt R.
:return: Nothing! The result is in T and U.
"""

def read_lattice(input_file=None):
    """
    Read a matrix from a file, or from stdin.
    :param input_file: file name, or when None, read from stdin.
    :return: a matrix consisting of column vectors.
    """
    data = []
    if input_file is None:
        data.append(input())
        while data[-1][-2] != ']':
            data.append(input())
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data.append(f.readline()[:-1])
            while data[-1] != ']' and data[-1][-2] != ']':
                data.append(f.readline()[:-1])

    # Strip away starting '[' and ending ']'
    assert data[0][0] == '[' and data[-1][-1] == ']'
    data[0] = data[0][1:]
    if data[-1] == ']':
        # Flatter and fpLLL output ']' on a separate line instead of '[<data of last row>]]'
        data.pop()
    else:
        data[-1] = data[-1][:-1]

    # Convert data to list of integers
    data = [list(map(int, line[1:-1].strip().split(' '))) for line in data]

    # Use column vectors.
    return np.ascontiguousarray(np.array(data, dtype=np.int64).transpose())

def batch_babai(R,T,U):
    return nearest_plane( R,T,U )

def assert_near_integers(W_cand: np.ndarray, *, atol=1e-8, rtol=0.0):
    W = np.asarray(W_cand, dtype=float)
    W_round = np.rint(W)  # nearest integer, stays float
    if not np.allclose(W, W_round, atol=atol, rtol=rtol, equal_nan=False):
        bad = np.flatnonzero(~np.isclose(W, W_round, atol=atol, rtol=rtol))
        i = bad[0]
        raise AssertionError(
            f"W_cand has non-integer entries (within tol). "
            f"Example idx={np.unravel_index(i, W.shape)}, "
            f"value={W.flat[i]!r}, nearest_int={W_round.flat[i]!r}, "
            f"abs_err={abs(W.flat[i]-W_round.flat[i]):.3e}"
        )

n = 144

# A = IntegerMatrix(n,n)
# A.randomize("qary", k=n//2, bits=14.2)

# t0 = perf_counter()
# LR = LatticeReduction( A )
# print(f"flatter done in: {perf_counter()-t0}")

# for beta in [14,40,24,30,66]:
#     t0 = perf_counter()
#     LR( cores=10, beta=beta, bkz_tours=2, verbose  = True )
#     print(f"BKZ-{beta} done in: {perf_counter()-t0}")
# for beta in range(40,43):
#     t0 = perf_counter()
#     LR( cores=10, beta=beta, bkz_tours=2, verbose  = True )
#     print(f"BKZ-{beta} done in: {perf_counter()-t0}")

# A = np.array( list(line for line in LR.B) ).transpose()
# print(A)
# write_lattice( A, output_file=f"lattst_{n}.txt" )

A = read_lattice( f"lattst_{n}.txt" )

n, _ = np.shape( A )
m =  144
cd = 50
start_at = n-cd

S = np.array([
    [ (i-j)%n for i in range(m)  ] for j in range(n)
])

E = np.array([
    [ randrange(-1,2) for i in range(m)  ] for j in range(n) #randrange(-2,3) 
])

lol = perf_counter()
# R = np.linalg.qr(A, mode='r') 
t0 = perf_counter()
Q, R = np.linalg.qr(A.astype(np.float64), mode='reduced') 
print(f"QR done in: {perf_counter()-t0}")
Qinv = np.linalg.inv(Q)

U =  np.zeros((n-start_at,m),dtype=np.int64) #np.zeros((n,m),dtype=np.int64)

lol = perf_counter()
# T = (A@S+E).astype(np.float64) #dim 512, 20k targets ~3 sec
T = ( R@S.astype(np.float64)+Qinv@E ) #dim 512, 20k targets ~0.3 sec
print(f"matmul-n-add: {perf_counter()-lol}")

Rcp = deepcopy(R)[start_at:,start_at:]
Tcp = deepcopy(T)[start_at:]


t0 = perf_counter()
nearest_plane(Rcp, Tcp, U)
U = -U
print(f"NP done in: {perf_counter()-t0}")

lines = ( T[start_at:] - R[start_at:,start_at:]@U.astype(np.float64) )
dR = np.diag(R)
for i in range( start_at,n ):
    assert all( np.abs(lines[i-start_at] / dR[i]) < 0.501 ), f"Noo: {lines[i-start_at] / dR[i]}"
print("- - - manual check")
W_cand = np.linalg.inv(R[-cd:,-cd:])@(T[start_at:]-lines)
print(np.round(W_cand))
assert_near_integers(W_cand)
print("- - -")

print("u:")
print(U)
print()
print( np.shape(lines) )

# Rcpp = deepcopy(R)[start_at:,start_at:]
# Tcpp = deepcopy(T)[start_at:] #!

# t0 = perf_counter()
# Ucpp = proj_submatrix_modulus_blas( Rcpp,Tcpp )
# print(f"Matmod done in: {perf_counter()-t0}")

# lines = ( T[start_at:] + R[start_at:,start_at:]@Ucpp.astype(np.float64) )
# dR = np.diag(R)
# for i in range( start_at,n ):
#     assert all( np.abs(lines[i-start_at] / dR[i]) < 0.501 ), f"Noo: {lines[i-start_at] / dR[i]}"
# print("- - - lines my")
# W_cand = np.linalg.inv(R[-cd:,-cd:])@(T[start_at:]-lines)
# print(np.round(W_cand))
# assert_near_integers(W_cand)
# print("- - -")

B = IntegerMatrix.from_matrix(A.T)
G = GSO.Mat(B,float_type="dd")
G.update_gso()

T_canon = A@S+E

W = []
for s in T_canon.T:
    w = G.babai(s)
    W.append( w )
W = np.array( W ).T

print("W:")
print(W[-cd:])
print("S: ")
print(S[-cd:])
print()
assert not np.any(U-S[-cd:])

