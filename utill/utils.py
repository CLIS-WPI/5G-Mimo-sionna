"""
Utility functions for dataset generation and signal processing
"""

import numpy as np

def db2lin(x):
    """
    Convert decibel values to linear scale.
    
    Parameters:
    -----------
    x : float or numpy.ndarray
        Value(s) in decibels
        
    Returns:
    --------
    float or numpy.ndarray
        Value(s) in linear scale
    """
    return 10.0**(x/10.0)

def lin2db(x):
    """
    Convert linear scale values to decibels.
    
    Parameters:
    -----------
    x : float or numpy.ndarray
        Value(s) in linear scale
        
    Returns:
    --------
    float or numpy.ndarray
        Value(s) in decibels
    
    Notes:
    ------
    - Handles zero values by replacing them with a small positive number
    - Applies np.log10 element-wise for array inputs
    """
    # Avoid log of zero by replacing zeros with a small number
    x = np.where(x == 0, 1e-20, x)
    return 10.0 * np.log10(x)

def calculate_path_loss(distance, frequency):
    """
    Calculate free space path loss.
    
    Parameters:
    -----------
    distance : float or numpy.ndarray
        Distance in meters
    frequency : float
        Carrier frequency in Hz
        
    Returns:
    --------
    float or numpy.ndarray
        Path loss in dB
    """
    c = 3e8  # speed of light in m/s
    return 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4*np.pi/c)

def snr_to_noise_power(snr_db, signal_power_db):
    """
    Convert SNR to noise power.
    
    Parameters:
    -----------
    snr_db : float or numpy.ndarray
        Signal-to-Noise Ratio in dB
    signal_power_db : float or numpy.ndarray
        Signal power in dB
        
    Returns:
    --------
    float or numpy.ndarray
        Noise power in linear scale
    """
    return db2lin(signal_power_db - snr_db)

def verify_power_levels(signal_power, noise_power, interference_power):
    """
    Verify that power levels are consistent.
    
    Parameters:
    -----------
    signal_power : numpy.ndarray
        Signal power in linear scale
    noise_power : numpy.ndarray
        Noise power in linear scale
    interference_power : numpy.ndarray
        Interference power in linear scale
        
    Returns:
    --------
    bool
        True if power levels are consistent, False otherwise
    """
    # Convert all powers to dB for comparison
    signal_db = lin2db(signal_power)
    noise_db = lin2db(noise_power)
    interference_db = lin2db(interference_power)
    
    # Check if signal power is greater than noise floor
    valid_signal = np.all(signal_db > noise_db)
    
    # Check if interference is within reasonable range
    valid_interference = np.all((interference_db > noise_db) & 
                            (interference_db < signal_db + 30))
    
    return valid_signal and valid_interference