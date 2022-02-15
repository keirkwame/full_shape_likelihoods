#!python3
import full_shape_likelihoods as fsl

def setup(options):
    """Setup likelihood class."""
    #Load options from block
    options_names = []
    options_dict = {options_name: options['full_shape_likelihoods', options_name] for options_name in options_names}

    # Initialise likelihood class
    likelihood_object = fsl.full_shape_spectra(options_dict)
    return likelihood_object

def execute(block, config):
    """Execute theory/likelihood calculation (galaxy clustering with BOSS data) for input linear cosmology."""
