# utill/utils.py

import numpy as np

def db2lin(x):
    """
    Convert from dB to linear scale
    
    Parameters
    ----------
    x : float or numpy.ndarray
        Value(s) in dB
        
    Returns
    -------
    float or numpy.ndarray
        Value(s) in linear scale
    """
    return 10.0**(x/10.0)

def lin2db(x):
    """
    Convert from linear scale to dB
    
    Parameters
    ----------
    x : float or numpy.ndarray
        Value(s) in linear scale
        
    Returns
    -------
    float or numpy.ndarray
        Value(s) in dB
    """
    return 10.0 * np.log10(x)
