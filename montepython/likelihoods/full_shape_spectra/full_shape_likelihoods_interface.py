#!python3
import numpy as np
import full_shape_likelihoods as fsl
from cosmosis.datablock import names

def setup(options):
    """Setup likelihood class."""
    #Load options from block
    options_names = ['nz', 'z', 'inv_nbar', 'no_wiggle', 'use_P', 'use_Q', 'use_B', 'use_AP', 'data_directory',
                     'P_measurements', 'B_measurements', 'AP_measurements', 'covmat_file', 'discreteness_weights_file',
                     'kminP', 'kmaxP', 'kmaxQ', 'kminB', 'kmaxB', 'ksizeB', 'bin_integration_P', 'h_fid', 'Hz_fid',
                     'DA_fid', 'rdHfid', 'rdDAfid']
    options_dict = {options_name: options['full_shape_likelihoods', options_name] for options_name in options_names}

    # Initialise likelihood class
    likelihood_object = fsl.full_shape_spectra(options_dict)
    return likelihood_object

def execute(block, config):
    """Execute theory/likelihood calculation (galaxy clustering with BOSS data) for input linear cosmology."""
    likelihood_object = config

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
    log_like = likelihood_object.loglkl(cosmo, nuisance_parameters)
    print('log_like =', log_like)
    block[names.likelihoods, 'full_shape_likelihoods_like'] = log_like

    return 0

def cleanup(config):
    """Cleanup."""
    pass
