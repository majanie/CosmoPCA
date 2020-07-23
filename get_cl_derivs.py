from astropy.io import ascii
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import numpy as np

info_txt = r"""
likelihood:
  planck_2018_lowl.TT:
  planck_2018_lowl.EE:
  planck_2018_highl_plik.TTTEEE:
  planck_2018_lensing.clik:
theory: 
  classy:
    extra_args: {N_ncdm: 1}
params:
  logA:
    prior: {min: 2, max: 4}
    ref: {dist: norm, loc: 3.05, scale: 0.001}
    proposal: 0.001
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  A_s: {value: 'lambda logA: 1e-10*np.exp(logA)', latex: 'A_\mathrm{s}'}
  theta_s_1e2:
    prior:
      min: 0.5
      max: 10
    ref:
      dist: norm
      loc: 1.0416
      scale: 0.0004
    proposal: 0.0002
    latex: 100\theta_\mathrm{s}
    drop: true
  100*theta_s:
    value: 'lambda theta_s_1e2: theta_s_1e2'
    derived: false
  n_s:
    prior: {min: 0.8, max: 1.2}
    ref: {dist: norm, loc: 0.96, scale: 0.004}
    proposal: 0.002
    latex: n_\mathrm{s}
  H0:
    latex: H_0
  omega_b:
    prior: {min: 0.005, max: 0.1}
    ref: {dist: norm, loc: 0.0221, scale: 0.0001}
    proposal: 0.0001
    latex: \Omega_\mathrm{b} h^2
  omega_cdm:
    prior: {min: 0.001, max: 0.99}
    ref: {dist: norm, loc: 0.12, scale: 0.001}
    proposal: 0.0005
    latex: \Omega_\mathrm{c} h^2
  Omega_k:
    prior:
      min: -0.3
      max: 0.3
    ref: 0.0
    latex: \Omega_K
  w0_fld:
    prior:
      min: -3.0
      max: -0.3
    ref: 0.0
    latex: w_0
  wa_fld:
    value: 0.0
  cs2_fld:
    value: 1
  Omega_Lambda:
    value: 0.0
  m_ncdm:
    prior:
      min: 0
      max: 5
    ref: 0.06
  N_ur:
    prior:
      min: 0.05
      max: 10.0
    ref: 1.0
  tau_reio:
    prior: {min: 0.01, max: 0.8}
    ref: {dist: norm, loc: 0.06, scale: 0.01}
    proposal: 0.005
    latex: \tau_\mathrm{reio}
"""

def initiate_model(info_text):
    info = yaml_load(info_txt)
    info['packages_path'] = '/home/moon/mniemeyer/cobaya_modules'
    model = get_model(info)
    point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))
    return model, point

def get_cls(omega_b, omega_cdm, logA, n_s, theta_s_1e2, tau_reio, Omega_k, m_ncdm, N_ur, w0_fld):
    point.update({'omega_b': omega_b, 'omega_cdm': omega_cdm, 'theta_s_1e2': theta_s_1e2,
              'logA': logA, 'n_s': n_s, 'tau_reio': tau_reio, 
             'Omega_k': Omega_k, 'm_ncdm': m_ncdm, 'w0_fld': w0_fld,
             'N_ur': N_ur})
    
    logposterior = model.logposterior(point)  # to force computation
    Cls = model.provider.get_Cl(ell_factor=True)
    return Cls

def get_Cl_param(value):
    params = {'omega_b': 0.022383, 'omega_cdm': 0.12011, 'theta_s_1e2': 1.040909,
              'logA': 3.0448, 'n_s': 0.96605, 'tau_reio': 0.0543, 
             'Omega_k': -0.011, 'm_ncdm': 0.06, 'w0_fld': -1.57,
             'N_ur': 2.89}
    params[param_name] = value
    Cls = get_cls(**params)
    print(param_name, params[param_name])
    return Cls

model, point = initiate_model(info_txt)

default = {'omega_b': 0.022383, 'omega_cdm': 0.12011, 'theta_s_1e2': 1.040909,
              'logA': 3.0448, 'n_s': 0.96605, 'tau_reio': 0.0543, 
             'Omega_k': -0.011, 'm_ncdm': 0.06, 'w0_fld': -1.57,
             'N_ur': 2.89}

steps = {'omega_b': 0.1*default["omega_b"],
         'omega_cdm': 0.1*default["omega_cdm"],
         'theta_s_1e2': 0.002, # HAPPY #  try more! probably small value, <= 0.1
         'logA': 0.1*default["logA"],
         'n_s': 0.1*default["n_s"],
         'tau_reio': 0.001, 
         'Omega_k': 0.04,  # HAPPY #  try more! probably around 1
         'm_ncdm': 0.06,    # try more! has to be larger than 0.06 because this is our default.
         'w0_fld': 0.8*1.2,  # try more! probably around 0.1 - 0.5, should be larger than 0.1...
         'N_ur': 0.1*1.2}

derivatives = {}

for param_name in default.keys():
    derivatives[param_name] = {}
    xval = default[param_name]
    step = steps[param_name] 
    up = xval + step
    h = up - xval
    low = xval - h
    upper = get_Cl_param(up)
    lower = get_Cl_param(low)
    ells = upper["ell"]
    for cl_key in ["tt", "te", "ee"]:
        derivatives[param_name][cl_key] = ( (upper[cl_key] - lower[cl_key])/(2*h) )
        derivatives[param_name][cl_key] *= 2*np.pi/(ells*(ells+1))
        ascii.write(derivatives[param_name], "derivatives/{}.tab".format(param_name), overwrite=True )