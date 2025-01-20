###################################################################################################################
#src/config.py
# Configuration File
# For Dataset Generation in Sionna based on Simulation Plan.txt
#####################################################################################################################
# Configuration File for Dataset Generation in Sionna
# This file contains the configurations necessary for generating simulated channel datasets
# using the Sionna library, which is based on the simulation plan described in "Simulation Plan.txt".
# The datasets are used for training and validating machine learning models in MIMO (Multiple Input, Multiple Output) systems.
#
# Purpose:
# This configuration file provides parameters for:
# 1. General settings such as the dataset size, random seed, and noise floor.
# 2. MIMO system settings including transmit and receive antenna configurations.
# 3. Resource grid specifications for generating OFDM symbols and subcarriers.
# 4. Channel model settings for simulating Rayleigh block fading channels and various propagation conditions.
# 5. Simulation settings for generating batches of channel realizations using Sionna.
#
# Inputs:
# - CONFIG: General configurations including dataset size and output directory.
# - MIMO_CONFIG: Defines the MIMO system setup such as the number of antennas, polarization, and spacing.
# - RESOURCE_GRID: Contains OFDM-related configurations such as subcarriers, bandwidth, and modulation format.
# - CHANNEL_CONFIG: Defines channel model parameters like the type of fading model, SNR range, and delay spread.
# - SIONNA_CONFIG: Specifies simulation parameters such as batch size and parallel workers.
#
# Outputs:
# - OUTPUT_FILES: Paths to the output files where the generated dataset will be saved. This includes:
#   - training_data.npy
#   - validation_data.npy
#   - test_data.npy
#
# This configuration allows for consistent dataset generation for training, validation, and testing
# in reinforcement learning and other machine learning-based beamforming optimization tasks.
#####################################################################################################################

#####################################################################################################################
import os
# General Configuration
CONFIG = {
    "random_seed": 42,  # Seed for reproducibility
    "output_dir": "./data/",  # Directory to store generated datasets
    "dataset_size": 1320000,  # Total number of samples to generate
    "validation_size": 100000,  # Number of validation samples
    "test_size": 100000,  # Number of test samples
    "noise_floor": -174,  # Noise floor in dBm/Hz
    "number_of_episodes": 1,  # Number of training episodes
    "mini_batch_size": 256,  # Mini-batch size for training
    "batch_size": 256,  # Add this line - Same as SIONNA_CONFIG batch_size
    "actor_lr": 1e-4,  # Learning rate for the actor network
    "critic_lr": 1e-3,  # Learning rate for the critic network
    "alpha_lr": 1e-4,  # Learning rate for the alpha parameter
    "validation_interval": 10,
    "checkpoint_interval": 10,  # Save checkpoint every 5 episodes
    "num_users": 4,  # Number of users in the system
    "interference_threshold": -90,  # Interference threshold in dBm
    "sinr_target": 20,  # Target SINR in dB

}

# MIMO Configuration
MIMO_CONFIG = {
    "tx_antennas": 4,  # Number of transmit antennas
    "rx_antennas": 4,  # Number of receive antennas
    "array_type": "ULA",  # Uniform Linear Array
    "element_spacing": 0.5,  # Spacing between elements in wavelengths
    "polarization": "single",  # Single polarization
    "antenna_gain": 15,  # Change from (5,15) to fixed 15 dBi for maximum gain
    "antenna_pattern": "directional",  # Beam pattern type
    "array_orientation": "fixed",  # Azimuth/elevation angles (static)
    "sinr_weight": 0.3,  # Weight for SINR in reward calculation
    "sinr_threshold": 10.0,  # Minimum acceptable SINR in dB
    "num_streams_per_tx": 1,  # Number of streams per transmitter
    "stream_management": True,  # Enable stream management
    "interference_coordination": True,  # Enable interference coordination
    "precoding_type": ["zf", "mmse", "mf"],  # Supported precoding types
}

# Resource Grid Configuration
RESOURCE_GRID = {
    "subcarriers": 64,  # fft_size
    "subcarrier_spacing": 30e3,  # In Hz
    "ofdm_symbols": 14,  # Number of OFDM symbols per slot
    "symbol_duration": 71.4e-6,  # Duration of one symbol in seconds
    "bandwidth": 2e6,  # Bandwidth in Hz
    "modulation_order": "QPSK",  # Modulation format
    "num_guard_carriers": (5, 6),  # Guard carriers left/right (based on typical LTE/5G settings)
    "dc_null": True  # Null the DC subcarrier
}


# Channel Model Configuration
CHANNEL_CONFIG = {
    "type": "rayleigh",  # Rayleigh block fading model
    "num_paths": 6,      # Reduce from 10 to 6 to focus on stronger paths
    "coherence_time": 1,  # Coherence time in slots
    "user_type": "static",  # Static user positions
    "path_loss_model": "FSPL",  # Free space path loss model
    "snr_range": (10, 30),  # Change from (0,30) to (10,30) for better SNR
    "delay_spread": (0.1e-6, 0.5e-6),  # Reduce max delay spread
    "doppler_shift": {
        "min_speed": 0.1,  # m/s (minimum speed to avoid purely static scenario)
        "max_speed": 30.0,  # m/s (typical pedestrian speed)
        "carrier_frequency": 3.5e9,  # Hz (carrier frequency in Hz)
        },
    "spatial_consistency": "static",  # No variations in channel
    "antenna_height": 1.5,  # Fixed antenna height in meters
    "interference": {
        "enabled": True,
        "model_type": "multi_user",
        "num_interferers": 2,        # Reduced from 3 to 2 interferers
        "interference_power_range": (-100, -80),  # Changed from (-120, -60) to (-100, -80)
        "spatial_correlation": 0.3,  # Reduced from 0.5 to 0.3
    },
    "sinr_calculation": {
        "enabled": True,
        "method": "exact",
        "averaging_window": 5,      # Reduced from 10 to 5
    },
}

MULTIUSER_CONFIG = {
    "enabled": True,
    "num_users": CONFIG["num_users"],
    "user_distribution": "random",  # How users are distributed
    "min_user_separation": 10,  # Minimum separation between users in meters
    "scheduling_type": "round_robin",  # User scheduling algorithm
    "fairness_metric": "proportional",  # Fairness metric for scheduling
}

# Sionna Simulation Settings
SIONNA_CONFIG = {
    "batch_size": 50,
    "num_realizations": CONFIG["dataset_size"],
    "num_workers": 4,
    "stream_management": True,  # Enable stream management in Sionna
    "interference_modeling": True,  # Enable interference modeling
    "precoding_enabled": True,  # Enable precoding
}

# Output Configuration
# At the beginning of the script, add:


# OUTPUT_FILES paths 
OUTPUT_FILES = {
    "training_data": os.path.join(".", "data", "training", "training_data.npy"),
    "validation_data": os.path.join(".", "data", "validation", "validation_data.npy"),
    "test_data": os.path.join(".", "data", "test", "test_data.npy")
}

# Debug Configuration
DEBUG = {
    "log_level": "INFO",  # Logging level (DEBUG, INFO, WARNING, ERROR)
    "save_intermediate": False,  # Save intermediate files for debugging
}

if __name__ == "__main__":
    print("Configuration loaded successfully.")
