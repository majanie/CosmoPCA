from astropy.io import ascii
import glob
import numpy as np

# read the Cl's
Cl = ascii.read("Cls/Cls_planck_best_fit.tab")
ells = Cl["ell"]
ell_here = (30 <= ells)&(ells <= 2000)
ells = ells[ell_here]
Cl = Cl[ell_here]
ell_factor = 2*np.pi/(ells*(ells+1)) # redo the ell_factor=True in the get_Cls
Cl["tt"] *= ell_factor
Cl["te"] *= ell_factor
Cl["ee"] *= ell_factor

f_sky = 0.8
f_l = Cl["ell"] / ((2*Cl["ell"]+1)*f_sky)

COV = {}
COV[("tt","tt")] = f_l * Cl["tt"]**2
COV[("ee","ee")] = f_l * Cl["ee"]**2
COV[("te","te")] = f_l * (Cl["te"]**2 + Cl["tt"] * Cl["ee"])
COV[("tt","ee")] = f_l * Cl["te"]**2
COV[("tt","te")] = f_l * Cl["te"] * Cl["tt"]
COV[("ee","te")] = f_l * Cl["te"] * Cl["ee"]

covmat = np.empty(shape=(3, 3, len(ells)), dtype=np.float64) # TT, TE, EE

covmat[0,0] = COV[("tt","tt")]
covmat[0,1] = COV[("tt","te")]
covmat[0,2] = COV[("tt","ee")]
covmat[1,0] = COV[("tt","te")]
covmat[1,1] = COV[("te","te")]
covmat[1,2] = COV[("ee","te")]
covmat[2,0] = COV[("tt","ee")]
covmat[2,1] = COV[("ee","te")]
covmat[2,2] = COV[("ee","ee")]

cov_inv = np.empty(covmat.shape, dtype=np.float64)
for i in range(covmat.shape[-1]):
    cov_inv[:,:,i] = np.linalg.inv(covmat[:,:,i])
    
# load derivatives
ff = glob.glob("derivatives/*")
derivs = {}
for fin in ff:
    param = fin.split("/")[1][:-4]
    derivs[param] = ascii.read(fin)[ell_here]
params = derivs.keys()

fischer = {}
for p_i in params:
    for p_j in params:
        sums = np.sum([[derivs[p_i][X]*cov_inv[i_x,i_y]*derivs[p_j][Y] for i_x,X in enumerate(["tt","te","ee"])]for i_y,Y in enumerate(["tt","te","ee"])])
        fischer[(p_i, p_j)] = sums
        
fischer_mat = np.empty((len(params), len(params)))
for i, p_i in enumerate(params):
    for j, p_j in enumerate(params):
        fischer_mat[i,j] = fischer[(p_i, p_j)]
        
fischer_inv = np.linalg.inv(fischer_mat)

fischer_dict = {}
for p in params:
    fischer_dict[p] = []
    for i, q in enumerate(params):
        fischer_dict[p].append(fischer[(p,q)])
        
ascii.write(fischer_dict, "fischer_matrix.tab")
print("Wrote to fischer_matrix.tab")

fischer_dict_inv = {}
for i, p in enumerate(params):
    fischer_dict_inv[p] = []
    for j, q in enumerate(params):
        fischer_dict_inv[p].append(fischer_inv[i,j])
        
ascii.write(fischer_dict_inv, "fischer_matrix_inverse.tab")
print("Wrote to fischer_matrix_inverse.tab")