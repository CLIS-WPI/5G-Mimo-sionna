#############################
# Configuration File
# For Dataset Generation in Sionna based on Simulation Plan.txt
#############################

# General Configuration
CONFIG = {
    "random_seed": 42,  # Seed for reproducibility
    "output_dir": "./data/",  # Directory to store generated datasets
    "dataset_size": 1320000,  # Total number of samples to generate
    "validation_size": 100000,  # Number of validation samples
    "test_size": 100000,  # Number of test samples
    "noise_floor": -174,  # Noise floor in dBm/Hz
}

# MIMO Configuration
MIMO_CONFIG = {
    "tx_antennas": 4,  # Number of transmit antennas
    "rx_antennas": 4,  # Number of receive antennas
    "array_type": "ULA",  # Uniform Linear Array
    "element_spacing": 0.5,  # Spacing between elements in wavelengths
    "polarization": "single",  # Single polarization
    "antenna_gain": (5, 15),  # Min and max gain in dBi
    "antenna_pattern": "directional",  # Beam pattern type
    "array_orientation": "fixed",  # Azimuth/elevation angles (static)
}

# Resource Grid Configuration
RESOURCE_GRID = {
    "subcarriers": 64,  # Number of subcarriers
    "subcarrier_spacing": 30e3,  # Subcarrier spacing in Hz
    "ofdm_symbols": 14,  # Number of OFDM symbols per slot
    "symbol_duration": 71.4e-6,  # Duration of one symbol in seconds
    "bandwidth": 2e6,  # Bandwidth in Hz
    "modulation_order": "QPSK",  # Modulation format
}

# Channel Model Configuration
CHANNEL_CONFIG = {
    "type": "rayleigh",  # Rayleigh block fading model
    "num_paths": 10,  # Number of multipath components
    "coherence_time": 1,  # Coherence time in slots
    "user_type": "static",  # Static user positions
    "path_loss_model": "FSPL",  # Free space path loss model
    "snr_range": (0, 30),  # SNR range in dB
    "delay_spread": (0.1e-6, 1e-6),  # Delay spread in seconds
    "doppler_shift": 0,  # Doppler shift for static scenario
    "spatial_consistency": "static",  # No variations in channel
    "antenna_height": 1.5,  # Fixed antenna height in meters
    "interference": None,  # No interference for initial setup
}

# Sionna Simulation Settings
SIONNA_CONFIG = {
    "batch_size": 256,  # Batch size for simulation
    "num_realizations": CONFIG["dataset_size"],  # Number of channel realizations
    "num_workers": 4,  # Number of workers for parallel processing
}

# Output Configuration
OUTPUT_FILES = {
    "training_data": f"{CONFIG['output_dir']}training/training_data.npy",
    "validation_data": f"{CONFIG['output_dir']}validation/validation_data.npy",
    "test_data": f"{CONFIG['output_dir']}test/test_data.npy",
}

# Debug Configuration
DEBUG = {
    "log_level": "INFO",  # Logging level (DEBUG, INFO, WARNING, ERROR)
    "save_intermediate": False,  # Save intermediate files for debugging
}

if __name__ == "__main__":
    print("Configuration loaded successfully.")
