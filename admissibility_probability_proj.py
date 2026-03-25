from lwe_gen import generateLWEInstances
from lattice_reduction import LatticeReduction
from size_reduction import nearest_plane
from fpylll import IntegerMatrix, GSO, FPLLL
FPLLL.set_precision(208)
from zgsa_fast import find_beta
import numpy as np

def project_onto_last(G,v,cd):
    assert cd <= G.d, f"Too marge dim {cd}>{G.d}"

    v_gh = np.asarray( G.from_canonical( v ) )
    v_gh[:-cd] = 0
    return np.asarray(  G.to_canonical( v_gh ) )

n, m, q = 130, 130, 18839 # 327689
# dist_s, dist_param_s, dist_e, dist_param_e = "ternary", 1./2, "discrete_gaussian", 1.5
dist_s, dist_param_s, dist_e, dist_param_e = "ternary", 1/3., "discrete_gaussian", 1.,
kappa = 20
n_targets = 500

a, b, n_dims = 50, n+m-kappa, 5 #(50, 100, 6) -> [50,60,...,100]
cds = np.asarray( np.round( np.linspace( a, b, n_dims ) ), dtype=int )
# cds=[50]

A, _, bse = generateLWEInstances(n, m, q, dist_s, dist_param_s, dist_e, dist_param_e, n_targets)
assert(len(bse)==n_targets)


#generate lwe basis
B = [ [int(0) for j in range(m+n)] for i in range(m+n) ]
for i in range(m):
    B[i][i] = int(q)
for i in range(m, m+n):
    B[i][i] = 1
for i in range(m, m+n):
    for j in range(m):
        B[i][j] = int( A[i-m,j] )

#split the basis
Htmp = B[:len(B)-kappa]
H = IntegerMatrix.from_matrix( [h[:len(B)-kappa] for h in Htmp] )
C = np.array( [b[:len(B)-kappa] for b in B[len(B)-kappa:]] )

#expected beta sufficient for successful babai for all targets under correct guesses
#might overestimate beta because our BDD error consists of two parts: Gaussian and ternary
beta = find_beta(n+m-kappa, n, q, 3*dist_param_e)
print("found beta:", beta)
if beta > n:
    beta = 50

bkz_tours = 5
lll_size = 64
LatRed_instance = LatticeReduction(H)
Hred = LatRed_instance(lll_size=lll_size, delta=0.99, cores=1, beta=42, bkz_tours=2)
print(f"BKZ-{42} done", flush=True) #faster to preprocess for beta > 30
beta = min(52,beta)
if beta>40:
    Hred = LatRed_instance(lll_size=lll_size, delta=0.99, cores=5, beta=beta, bkz_tours=bkz_tours) 
    print(f"BKZ-{beta} done")

G = GSO.Mat( IntegerMatrix.from_matrix( Hred ), float_type="mpfr")
G.update_gso()

babai_lift_success = 0
for i in range(len(bse) - 1, -1, -1):
    b, s, e = bse[i]

    sguess = s[-kappa:]
    sguess_times_C = sguess @ C
    target = np.concatenate([b, (n-kappa)*[0]]) - sguess_times_C
    babai_res = G.babai(target)
    tshift = target - G.B.multiply_left(babai_res)

    diff = tshift - np.concatenate([e, -s[:-kappa]])
    if np.all(np.isclose(diff, 0.0, atol=1e-7)):
        babai_lift_success += 1
    else:
        del bse[i] #we will not launch mitm on failed instances
        

print(f"{babai_lift_success} left")


# all_guesses_babied = False
results = {}

for j, cd in enumerate(cds):
    print( f"computing dim {cd}" )
    full_dim_succ = 0
    cd_dim_succ = 0
    for i, bse_inst in enumerate(bse[:]):
        b,s,e = bse_inst

        sguess = s[-kappa:]

        #check admissibility; Odlyzko-style splitting: sguess = [ sguess[:kappa/2] | 0..0 ] + [0..0 | sguess[kappa/2:]]
        #w = sguess[]
        #TODO: split s as above, run NP on these splits mulptiplied by C, check that the difference/sum is equal to tshift (which is equal to the error)
        w1 = np.concatenate([sguess[:kappa//2],(kappa//2)*[0]])
        w2 = np.concatenate([(kappa//2)*[0], sguess[kappa//2:]])

        target_w1 = np.concatenate( [b,(n-kappa)*[0]] ) - w1@C
        target_w2 =  -w2@C
        #same but in projective lattice
        target_w1_proj = project_onto_last(G,target_w1,cd)
        target_w2_proj = project_onto_last(G,target_w2,cd)

        tmp = target_w1@(target_w1-target_w1_proj)
        assert np.abs(tmp)

        babai_res_w1 = G.babai(target_w1)
        err_w1 = target_w1 - G.B.multiply_left(babai_res_w1)
        #same but in projective lattice
        babai_res_w1_proj = G.babai(target_w1_proj) #,start=G.d-cd
        err_w1_proj  = target_w1_proj  - G.B.multiply_left(babai_res_w1_proj)

        babai_res_w2 = G.babai(target_w2)
        err_w2 = target_w2 - G.B.multiply_left(babai_res_w2)
        #same but in projective lattice
        babai_res_w2_proj = G.babai(target_w2_proj) #,start=G.d-cd
        err_w2_proj = target_w2_proj - G.B.multiply_left(babai_res_w2_proj)

        err_w1_gs, err_w2_gs = G.from_canonical(err_w1), G.from_canonical(err_w2)
        lhs, rhs = G.from_canonical(np.concatenate([e,-s[:-kappa]])) - np.array( err_w1_gs ) , np.array( err_w2_gs )
        #same but in projective lattice #debug -- should be == lhs_proj, rhs_proj below
        lhs_proj2, rhs_proj2 = G.from_canonical(np.concatenate([e,-s[:-kappa]]))[-cd:] - np.array( err_w1_gs )[-cd:] , np.array( err_w2_gs )[-cd:]

        err_w1_proj, err_w2_proj = G.from_canonical(err_w1_proj)[-cd:] , G.from_canonical(err_w2_proj)[-cd:]
        lhs_proj, rhs_proj = G.from_canonical(np.concatenate([e,-s[:-kappa]]))[-cd:] - np.array( err_w1_proj ) , np.array( err_w2_proj )

        # print(f"lhs_proj1-lhs_proj: {lhs_proj2-lhs_proj}")
        # print(f"rhs_proj2-rhs_proj: {rhs_proj2-rhs_proj}")
        assert np.all( np.isclose(lhs_proj2-lhs_proj,0.0, atol=1e-7) ), f"lhs_proj2 != lhs_proj"
        assert np.all( np.isclose(rhs_proj2-rhs_proj,0.0, atol=1e-7) ), f"rhs_proj2 != rhs_proj"
        # print("- - -")
        
        diff = lhs-rhs
        # print(diff)
        if np.all( np.isclose(diff, 0.0, atol=1e-7) ):
            full_dim_succ+=1
        diff_proj = lhs_proj-rhs_proj
        if np.all( np.isclose(diff_proj, 0.0, atol=1e-7) ):
            cd_dim_succ+=1
        
        # error = G.from_canonical(np.concatenate([e,-s[:-kappa]]))
        # print( f"err_w1_gs: {err_w1_gs }" )
        # print( f"err_w2_gs: {err_w2_gs}" )
        # print(f"sum: {np.array(error)-np.array(err_w1_gs)}") #+err_w2_gs
        # print("- - -")

        # babai_res_check = G.babai(target_w1+target_w2)
        # tshift_check = target_w1+target_w2 - G.B.multiply_left(babai_res_check)
        # print(babai_res_check)

        # #same but in projective lattice
        # babai_res_check_proj = G.babai(target_w1_proj+target_w2_proj)
        # tshift_check_proj = target_w1_proj+target_w2_proj - G.B.multiply_left(babai_res_check_proj)
        # tshift_check_proj = project_onto_last(G,tshift_check_proj,cd)

        # corr_chk = np.concatenate([e,-s[:-kappa]]) - tshift_check #should be zero, if we were right
        # print( f"corr: {corr_chk}" )
        # if all( np.isclose(corr_chk, 0.0, atol=1e-7) ):
        #     full_dim_succ+=1

        # e_s_proj = project_onto_last(G, np.concatenate([e,-s[:-kappa]]), cd)
        # corr_chk_proj = e_s_proj - tshift_check_proj
        # print( f"corr_proj: {corr_chk_proj}" )
        # print(f"{(tshift_check@tshift_check)**0.5} | {(tshift_check_proj@tshift_check_proj)**0.5}")

        # if all( np.isclose(corr_chk_proj, 0.0, atol=2e-7) ):
        #     cd_dim_succ+=1
        #     assert np.all( np.abs(err_w1_gs) < 0.5), f"What?!"
        #     assert np.all( np.abs(err_w2_gs) < 0.5), f"What?!"
        #     assert np.all( np.abs(err_w1_gs+err_w2_gs) < 0.5), f"What?!"
        # print("-----------------------------------")


    print(f"{full_dim_succ} vs {cd_dim_succ} out of {babai_lift_success}")
    results[cd] = ( full_dim_succ, cd_dim_succ, babai_lift_success )

print(flush=True)
print(results)

print(A)