import numpy as np
import pickle

infdiff_correct= []
infdiff_incorrect= []

with open("res_140.pkl","rb") as file: #res1411
    results = pickle.load( file )

n=results[0]["n"]
kappa = results[0]["kappa"]
cd = results[0]["cd"]

l = []
adm_nums = []
for D in results:
    infdiff_correct = np.concatenate( [infdiff_correct,D['infdiff_correct']] )
    infdiff_incorrect = np.concatenate( [infdiff_incorrect,D['infdiff_incorrect']] )
    adm_nums.append(np.mean(D['is_adm_num_incorrect'])/D['n_trials'])

print(f"avg adm prob: {np.mean(adm_nums)}")

h = histogram(infdiff_correct, bins=20, color="green", alpha=0.65, label="Correct unif. pair") + histogram(infdiff_incorrect, bins=20, color="red", alpha=0.65,
                                                                          title=f"Infinity norm. n={n}, kappa={kappa}, cd={cd}", label="Wrong guess")

h.set_legend_options(handlelength=1)
h.legend(True)
h.show()
