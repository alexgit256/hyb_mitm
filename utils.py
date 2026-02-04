import sys, os
import glob #for automated search in subfolders
import numpy as np
from fpylll import GSO, IntegerMatrix, FPLLL
from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from math import sqrt, ceil, floor, log, exp
from copy import deepcopy

from blaster_core import \
    set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, ZZ_right_matmul
from size_reduction import nearest_plane

# try:
#     from multiprocess import Pool  # you might need pip install multiprocess
# except ModuleNotFoundError:
#     from multiprocessing import Pool

DTYPE = "double"


def gsomat_copy(M):
    n,m,int_type,float_type = M.B.nrows,M.B.ncols,M.int_type,M.float_type

    B = []
    for i in range(n):
        B.append([])
        for j in range(m):
            B[-1].append(int(M.B[i][j]))
    B = IntegerMatrix.from_matrix( B,int_type=int_type )

    U = []
    for i in range(n):
        U.append([])
        for j in range(m):
            U[-1].append(int(M.U[i][j]))
    U = IntegerMatrix.from_matrix( U,int_type=int_type )

    UinvT = []
    for i in range(n):
        UinvT.append([])
        for j in range(m):
            UinvT[-1].append(int(M.UinvT[i][j]))
    UinvT = IntegerMatrix.from_matrix( UinvT,int_type=int_type )

    M = GSO.Mat( B, float_type=float_type, U=U, UinvT=UinvT )
    M.update_gso()
    return M

def to_canonical_scaled(M, t, offset=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    # assert not( scale_fact is None ), "scale_fact is None "
    if len(t)==0:
        return np.array([])
    if offset is None:
        offset=M.d

    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r()[-offset:])
    r_ = np.array( [sqrt(scale_fact/tt) for tt in M.r()[-offset:]], dtype=DTYPE )
    tmp = t*r_
    return np.array( M.to_canonical(tmp, start=M.d-offset) )

def from_canonical_scaled(M, t, offset=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    # assert not( scale_fact is None ), "scale_fact is None "
    if len(t)==0:
        return np.array([])
    if offset is None:
        offset=M.d
    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r()[-offset:])
    t_ = np.array( M.from_canonical(t)[-offset:], dtype=DTYPE )
    r_ = np.array( [sqrt(tt/scale_fact) for tt in M.r()[-offset:]], dtype=DTYPE )

    return t_*r_

def to_canonical_scaled_start(M, t, dim=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of first coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if len(t)==0:
        return np.array([])
    if dim is None:
        dim=M.d
    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r()[:dim])
    r_ = np.array( [sqrt(scale_fact/tt) for tt in M.r()[:dim]], dtype=DTYPE )
    tmp = np.concatenate( [t*r_, (M.d-dim)*[0]] )

    return np.array( M.to_canonical(tmp,start=0) )

def from_canonical_scaled_start(M, t, dim=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of first coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if len(t)==0:
        return np.array([])
    if dim is None:
        dim=M.d
    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r()[:dim])
    t_ = np.array( M.from_canonical(t)[:dim], dtype=DTYPE )
    r_ = np.array( [sqrt(tt/scale_fact) for tt in M.r()[:dim]], dtype=DTYPE )

    return t_*r_

# def proj_submatrix_modulus(G, v, dim=None):
#     """
#     """
#     if dim is None:
#         dim = G.B.nrows
#     v_gh = (G.B.nrows-dim)*[0] + G.from_canonical(v)[-dim:]
#     c = G.babai(v,start=G.d-dim, gso=True)
#     v = G.to_canonical( (G.d-dim)*[0] + list(v_gh) )
    
#     Bsub = IntegerMatrix.from_matrix( [G.B[i] for i in range(G.d-dim,G.d)] )
#     shift = Bsub.multiply_left( c )
#     shift_gh = G.from_canonical(shift)[-dim:]
#     shift = G.to_canonical((G.d-dim)*[0] + list(shift_gh))

#     return np.array( v ) - np.array( shift )

def proj_submatrix_modulus(G, v, dim=None, coords_too=False):
    """
    Given GSO object G and target vector v:
    1) Projects v onto last dim GS vectors of G.B
    2) Finds vector bab_gh from Lat(last dim GS) close to v with Babai
    3) Returns v - bab_gh
    """
    d = G.B.nrows
    if dim is None:
        dim = d
    elif not (1 <= dim <= d):
        raise ValueError("dim must be in [1, G.B.nrows]")

    v_gh = (G.B.nrows-dim)*[0] + list( G.from_canonical(v,start=d-dim) ) #project v onto last dim GS vectors
    v_gh = np.asarray( G.to_canonical( v_gh ) ) #go back to canonical
    c = G.babai(v, gso=False, start=d-dim, dimension=dim) #do babai and take last dim coords (TODO: wth does "start" keyword mean?)
    bab_gh = (G.B.nrows-dim)*[0] + list( G.from_canonical( G.B[-dim:].multiply_left(c) )[-dim:] ) #project the close vec onto last dim GS vectors
    bab_gh = G.to_canonical( bab_gh )
    v_gh -= np.asarray( bab_gh ) #substract close lat vect from proj of v

    if coords_too:
          return v_gh, c
    return v_gh

def proj_submatrix_modulus_partenum(G, v, dim=None, coords_too=False):
    """
    Given GSO object G and target vector v:
    1) Projects v onto last dim GS vectors of G.B
    2) Finds vector bab_gh from Lat(last dim GS) close to v with partial enumeration + Babai
    3) Returns v - bab_gh
    """
    if dim is None:
        dim = G.B.nrows
    v_gh = (G.B.nrows-dim)*[0] + list( G.from_canonical(v)[-dim:] ) #project v onto last dim GS vectors
    v_gh = np.asarray( G.to_canonical( v_gh ) ) #go back to canonical
    c = G.babai(v, gso=False)[-dim:] #do babai and take last dim coords (TODO: wth does "start" keyword mean?)
    bab_gh = (G.B.nrows-dim)*[0] + list( G.from_canonical( G.B[-dim:].multiply_left(c) )[-dim:] ) #project the close vec onto last dim GS vectors
    bab_gh = G.to_canonical( bab_gh )
    v_gh -= np.asarray( bab_gh ) #substract close lat vect from proj of v

    if coords_too:
          return v_gh, c
    return v_gh

def reduce_to_fund_par_proj(B_gs,t_gs,dim):
    t_gs_save = deepcopy( t_gs )
    c = [0 for i in range(dim)]
    # for i in range(dim):
    for j in range(dim-1,-1,-1):
        mu = round( t_gs[j] / B_gs[j][j] )
        t_gs -= B_gs[j] * mu
        c[j] -= mu
    for i in range(dim):
        t_gs_save += c[i] * B_gs[i]
    return t_gs_save

# https://math.stackexchange.com/questions/4705204/uniformly-sampling-from-a-high-dimensional-unit-sphere
def random_on_sphere(d,r):
    """
    d - dimension of vector
    r - radius of the sphere
    """
    u = np.random.normal(0,1,d)  # an array of d normally distributed random variables
    d=np.sum(u**2) **(0.5)
    return r*u/d

# Borrowed from https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def uniform_in_ball(num_points, dimension, radius=1):
    from numpy import random, linalg
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T

def test_vect_proj( G, n_slicer_coord, n_tests, dist ):
    # Gives norms of n_tests projected and scaled vectors ~Bin(eta). The projection is onto
    # the last n_slicer_coord dimensional projective lattice.
    # dist = centeredBinomial(eta)

    gh_sub = gaussian_heuristic( G.r()[-n_slicer_coord:] )
    lens = []
    for cntr in range(n_tests):
        v = dist.sample(G.d)
        v_ = from_canonical_scaled( G, v, offset=n_slicer_coord,scale_fact=gh_sub )
        lv_ = (v_@v_)**0.5
        lens.append(lv_)
    return(lens)

def dist_babai(G, t):
    #Given GSO object G, returns distance between t and G.babai(t).
    cv = G.babai( t )
    v = np.array( G.B.multiply_left( cv ) )
    dist = (t-v)
    dist = (dist@dist)**0.5
    return dist

def find_vect_in_list(v,l,tolerance=1.0e-6):
    assert len(v) == len(l[0]), f"Shapes do not allign! {len(v)} vs. {len(l[0])}"
    mindiff = float("inf")
    # print(f"debug v: {v}")
    for i in range(len(l)):
        # print(f"debug ti: {l[i]}")
        tmp = np.abs( np.array(v)-np.array(l[i]) )
        # print(f"tmp: {tmp}")
        mindiff = min( mindiff, max(tmp) )
        if (mindiff<tolerance):
            # print(f"mindiff: {mindiff}")
            return i
    print(f"FAIL mindiff: {mindiff}")
    return None

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

    Tproj = T[d-dim:] #projection of arbitrary many columns of T onto the last dim coords
    Rproj = R[d-dim:,d-dim:] #R-factor of the last dim dimensional projective sublattice
    U = np.zeros( (dim,n),dtype=np.int64 )
    print(f"T: {np.shape(T)}")
    print(f"R: {np.shape(R)}")
    print(f"U: {np.shape(U)}")

    nearest_plane(R,T,U)

    return U
    