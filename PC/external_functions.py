import numpy as np

P = np.array([[-9.95652961e-01, -8.87260307e-02, -2.00566550e-02,
         8.56479637e-03, -7.74596995e-03, -1.61920309e-02,
        -4.51212756e-04, -1.45655039e-04,  2.20021570e-03],
       [-2.29320737e-02,  2.30293289e-02,  9.97377314e-01,
        -2.58002987e-02, -3.13337103e-02,  5.02416308e-02,
         6.27163961e-04, -2.77153658e-05,  3.19238342e-03],
       [-7.65714918e-02,  7.38829931e-01, -5.78746160e-02,
        -3.01647829e-01, -4.34757774e-02,  5.93272952e-01,
        -7.76656260e-03, -6.52209713e-04,  1.60354523e-03],
       [-4.66285843e-02,  6.09463045e-01,  3.58600971e-02,
         3.03252052e-01,  4.48394252e-01, -5.74715712e-01,
        -2.57905372e-02, -2.13227844e-03, -3.32801111e-02],
       [ 1.01250699e-02, -2.69687253e-01,  1.40486674e-02,
        -1.86786205e-01,  8.91081653e-01,  3.09493126e-01,
         4.48309668e-02,  7.95024095e-03, -1.39387415e-02],
       [ 2.88859175e-03, -3.63594348e-02,  8.43641557e-04,
         3.08347838e-01,  3.38546003e-02,  1.92801823e-01,
        -9.25730018e-01, -9.05886332e-02, -9.46057591e-03],
       [-4.00265578e-04, -5.70801601e-03,  5.68494654e-04,
         7.56013811e-01, -3.60328160e-03,  3.96706605e-01,
         3.47269603e-01, -9.07922103e-02, -3.77069671e-01],
       [-9.23027528e-04, -1.39953598e-02,  1.55129407e-03,
        -3.17790732e-01, -2.86645732e-02, -1.45393194e-01,
        -1.32649893e-01,  5.59286202e-02, -9.25270273e-01],
       [ 9.95464271e-05, -8.92769810e-04,  7.04368901e-06,
        -1.17441312e-01,  1.83380492e-03, -5.88965400e-02,
         4.57807693e-02, -9.90128407e-01, -1.68648720e-02]])
P_T = P.T

def get_omega_b(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[0] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_omega_cdm(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[1] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_theta_s_1e2(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[2] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_logA(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[3] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_A_s(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return 10**(-10)*np.exp(get_logA(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9))

def get_n_s(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[4] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_tau_reio(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[5] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_Omega_k(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[6] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_m_ncdm(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[7] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def get_N_ur(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    return (P_T[8] @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9]))

def external_prior(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9):
    priors = [[0.005, 0.1], # omega_b, y
             [0.001, 0.99], # omega_cdm, y
             [0.5, 10.0],   # theta_s, y
             [1.61, 3.91],    # logA, 1.61 to 3.91 # 2.7, 4.0
             [0.9, 1.1],    # ns, 0.8 to 1.2
             [0.01, 0.8],   # tau, y
             [-0.3, 0.3],   # Omega_k
             [0, 5],        # mnu
             #[-3.0, -0.3],  # w0
             [0.05, 10]]    # Nur

    within = 1
    params = P_T @ np.array([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9])
    for i in range(len(params)):
        within *= priors[i][0]<=params[i]<=priors[i][1]
    return within
