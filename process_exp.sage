import numpy as np
import pickle

def clip_percentile(x, lo=0, hi=99):
    lo_v, hi_v = np.percentile(x, [lo, hi])
    return [v for v in x if lo_v <= v <= hi_v]

infdiff_correct= []
infdiff_incorrect= []

with open("exp_256_156_28.pkl","rb") as file: #res_2026-02-05 163304
    results = pickle.load( file )

n=results[0]["n"]
kappa = results[0]["kappa"]
cd = results[0]["cd"]

l = []
adm_nums = []
infdiff_correct = []
infdiff_incorrect  = []
for D in results:
    infdiff_correct = np.concatenate( [infdiff_correct,D['infdiff_correct']] )
    infdiff_incorrect = np.concatenate( [infdiff_incorrect,D['infdiff_incorrect']] )
    adm_nums.append(np.mean(D['is_adm_num_incorrect'])/D['n_trials'])

perc = 88.
infdiff_correct_f = clip_percentile(infdiff_correct, 0, perc)
infdiff_incorrect_f = clip_percentile(infdiff_incorrect, 0, perc)
print(f"avg adm prob: {np.mean(adm_nums)}")

allv = infdiff_correct_f + infdiff_incorrect_f
bins = np.linspace(min(allv), max(allv), 129)

h = histogram(infdiff_incorrect_f, bins=bins, color="red", alpha=0.65,title=f"Infinity norm. n={n}, kappa={kappa}, cd={cd}", label="Wrong guess")
h += histogram(infdiff_correct_f, bins=bins, color="green", alpha=0.6, label="Correct unif. pair") 

h.set_legend_options(handlelength=0.5)
h.legend(True)
h.save(f"infnrm_{n}.png")
