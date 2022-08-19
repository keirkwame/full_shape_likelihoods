#!python3
import numpy as np
import ctypes
import gc
import json
import full_shape_likelihoods as fsl
from memprof import memprof
from cosmosis.datablock import names

def setup(options):
    """Setup likelihood class."""
    #Load options from block
    #options_names = ['nz', 'z', 'inv_nbar', 'no_wiggle', 'use_P', 'use_Q', 'use_B', 'use_AP', 'data_directory',
    #                 'P_measurements', 'B_measurements', 'AP_measurements', 'covmat_file', 'discreteness_weights_file',
    #                 'kminP', 'kmaxP', 'kmaxQ', 'kminB', 'kmaxB', 'ksizeB', 'bin_integration_P', 'h_fid', 'Hz_fid',
    #                 'DA_fid', 'rdHfid', 'rdDAfid']
    #options_dict = {options_name: options['full_shape_likelihoods', options_name] for options_name in options_names}
    params_json = options['full_shape_likelihoods', 'parameters_json']
    with open(params_json) as json_data:
        options_dict = json.load(json_data)

    # Initialise likelihood class
    likelihood_object = fsl.full_shape_spectra(options_dict)
    return likelihood_object

#@memprof(plot=True)
def execute(block, config):
    """Execute theory/likelihood calculation (galaxy clustering with BOSS data) for input linear cosmology."""
    likelihood_object = config

    #Get cosmological parameters
    h = block['cosmological_parameters', 'h0']
    A_s = block['cosmological_parameters', 'a_s']
    n_s = block['cosmological_parameters', 'n_s']

    #Get cosmological distances
    rs_drag = block['distances', 'rs_zdrag']
    z_distance = block['distances', 'z']
    h_z = block['distances', 'h']
    d_a = block['distances', 'd_a'] #Mpc

    #Get logarithmic growth rate
    k_growth, z_growth, f = block.get_grid('linear_cdm_transfer', 'k_h', 'z', 'growth_factor_f') #h/Mpc
    delta_tot = block.get('linear_cdm_transfer', 'delta_total')

    #Get linear matter power spectrum
    k_power, z_power, pk = block.get_grid('matter_power_lin', 'k_h', 'z', 'p_k') #Check order #h/Mpc, (Mpc/h)^3
    print('Matter power:', k_power.shape, z_power.shape, pk.shape)

    #Get cosmology
    cosmology = [(h, A_s, n_s), (rs_drag, z_distance, h_z, d_a), (k_growth, z_growth, f, delta_tot),
                 (k_power, z_power, pk)]

    #Get nuisance parameters
    nuisance_parameters = [None,] * 6
    nuisance_parameters[0] = np.array([None,] * likelihood_object.nz)
    nuisance_parameters[1] = np.array([None,] * likelihood_object.nz)
    nuisance_parameters[2] = np.array([None,] * likelihood_object.nz)
    nuisance_parameters[3] = block['full_shape_likelihoods', 'fNL_eq']
    nuisance_parameters[4] = block['full_shape_likelihoods', 'fNL_orth']
    nuisance_parameters[5] = block['full_shape_likelihoods', 'alpha_rs']

    for i in range(likelihood_object.nz):
        nuisance_parameters[0][i] = block['full_shape_likelihoods', 'b1_z%i'%(i+1)]
        nuisance_parameters[1][i] = block['full_shape_likelihoods', 'b2_z%i' % (i + 1)]
        nuisance_parameters[2][i] = block['full_shape_likelihoods', 'bG2_z%i' % (i + 1)]

    #Get log-likelihood
    log_like, kP, P0_theory, P2_theory, P4_theory, kQ, Q_theory, kB, B_theory, AP_theory = likelihood_object.loglkl(cosmology, nuisance_parameters)
    #(-600., np.linspace(0.0125, 0.1975, 20), np.zeros((20, 4)), np.zeros((20, 4)), np.zeros((20, 4)), np.linspace(0.2025, 0.3975, 20), np.zeros((20, 4)), np.linspace(0.015, 0.075, 7), np.zeros((80, 4)), np.zeros((2, 4)))
    #likelihood_object.loglkl(
    #    cosmology, nuisance_parameters)
    print('log_like =', log_like)
    block.put_grid('full_shape_likelihoods', 'k_multipoles', kP, 'z_multipoles',
                   likelihood_object.z[:likelihood_object.nz], 'monopole', P0_theory)
    block.put('full_shape_likelihoods', 'quadrupole', P2_theory)
    block.put('full_shape_likelihoods', 'hexadecapole', P4_theory)
    block.put_grid('full_shape_likelihoods', 'k_Q', kQ, 'z_Q', likelihood_object.z[:likelihood_object.nz], 'Q',
                   Q_theory)
    #print(B_theory, B_theory.shape, np.arange(B_theory.shape[0]))
    block.put_grid('full_shape_likelihoods', 'index_B', np.arange(B_theory.shape[0]), 'z_B', likelihood_object.z[:likelihood_object.nz], 'B',
                   B_theory)
    block.put('full_shape_likelihoods', 'k_B', kB)
    block.put('full_shape_likelihoods', 'Alcock_Paczynski', AP_theory)
    if np.isnan(log_like):
        log_like = -1.e+30
    block[names.likelihoods, 'full_shape_likelihoods_like'] = log_like

    #Delete memory
    del(config)
    del(cosmology)
    del(likelihood_object)
    gc.collect()

    #libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")
    #libc.malloc_trim(0)

    return 0

def cleanup(config):
    """Cleanup."""
    del(config)
    gc.collect()

    #libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")
    #libc.malloc_trim(0)

    return 0
