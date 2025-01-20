#############################
#/src/dataset_generator.py
# Dataset Generation Script for Simulated MIMO Channel Realizations
# This script generates synthetic datasets for training, validation, and testing
# of machine learning models used in MIMO (Multiple Input, Multiple Output) systems. 
# The dataset consists of channel realizations and corresponding Signal-to-Noise Ratio (SNR) values 
# for different simulation conditions. The generated datasets are used for beamforming optimization 
# tasks and reinforcement learning in the context of wireless communication systems.

# Purpose:
# This script performs the following tasks:
# 1. Generates random SNR values within a specified range.
# 2. Simulates MIMO channel realizations using a Rayleigh block fading model with predefined MIMO antenna configurations.
# 3. Creates separate datasets for training, validation, and testing.
# 4. Saves the datasets to the specified output files for later use in training machine learning models.
#
# Inputs:
# - CONFIG: General configuration for dataset size, output directory, and noise floor.
# - MIMO_CONFIG: MIMO antenna configuration, including the number of transmit and receive antennas, polarization, and spacing.
# - RESOURCE_GRID: Resource grid configuration, including subcarriers, bandwidth, and modulation format.
# - CHANNEL_CONFIG: Channel model configuration, including SNR range, fading model, and delay spread.
# - SIONNA_CONFIG: Simulation settings including batch size and the number of realizations.
#
# Outputs:
# - The script generates and saves datasets for training, validation, and testing:
#   - channel_realizations: Simulated channel realizations (Rayleigh fading channels) for each sample.
#   - snr: Corresponding Signal-to-Noise Ratio values for each realization.
#   - The datasets are saved as `.npy` files at the paths specified in the OUTPUT_FILES dictionary.
#
# The script ensures that the required directories exist before saving the datasets and processes the data 
# in batches to improve memory efficiency during simulation.

#############################
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
from sionna.channel import OFDMChannel, RayleighBlockFading
from sionna.channel.tr38901 import AntennaArray
from sionna.mimo import StreamManagement
from sionna.mimo.precoding import zero_forcing_precoder, normalize_precoding_power
from utill.utils import db2lin, lin2db
from config import CONFIG, MIMO_CONFIG, RESOURCE_GRID, CHANNEL_CONFIG, SIONNA_CONFIG, OUTPUT_FILES

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_FILES["training_data"]), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILES["validation_data"]), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILES["test_data"]), exist_ok=True)

def create_antenna_array():
    tx_array = AntennaArray(
        num_rows=1,
        num_cols=MIMO_CONFIG["tx_antennas"],
        polarization=MIMO_CONFIG["polarization"],
        polarization_type="V",  
        antenna_pattern="38.901",  
        carrier_frequency=RESOURCE_GRID["bandwidth"],
        vertical_spacing=MIMO_CONFIG["element_spacing"],
        horizontal_spacing=MIMO_CONFIG["element_spacing"],
    )
    rx_array = AntennaArray(
        num_rows=1,
        num_cols=MIMO_CONFIG["rx_antennas"],
        polarization=MIMO_CONFIG["polarization"],
        polarization_type="V",  
        antenna_pattern="38.901",  
        carrier_frequency=RESOURCE_GRID["bandwidth"],
        vertical_spacing=MIMO_CONFIG["element_spacing"],
        horizontal_spacing=MIMO_CONFIG["element_spacing"],
    )
    return tx_array, rx_array

def create_stream_management(num_users):
    """Improved stream management following Sionna's implementation"""
    # Create association matrix
    rx_tx_association = np.zeros([num_users, MIMO_CONFIG["tx_antennas"]])
    
    # Ensure proper stream allocation
    streams_per_user = MIMO_CONFIG["num_streams_per_tx"]
    total_streams = min(MIMO_CONFIG["tx_antennas"], num_users * streams_per_user)
    
    for i in range(total_streams):
        user_idx = i % num_users
        antenna_idx = i % MIMO_CONFIG["tx_antennas"]
        rx_tx_association[user_idx, antenna_idx] = 1
    
    return StreamManagement(
        rx_tx_association, 
        num_streams_per_tx=streams_per_user
    )

def calculate_sinr(desired_signal, interference_signals, noise_power):
    """Improved SINR calculation following Sionna's approach"""
    # Convert all values to linear scale first
    signal_power = np.abs(desired_signal)**2
    
    # Apply antenna gain properly
    antenna_gain_linear = db2lin(MIMO_CONFIG["antenna_gain"])
    signal_power *= antenna_gain_linear
    
    # Calculate interference including both inter-user interference and noise
    total_interference = interference_signals + noise_power
    
    # Calculate SINR
    sinr_linear = signal_power / total_interference
    
    # Convert to dB and clip to reasonable range
    sinr_db = lin2db(sinr_linear)
    return np.clip(sinr_db, -20, 30)  # Clip to reasonable range

def create_channel_model(num_users):
    """Create channel model with proper number of users"""
    # Current implementation is too simplified
    # Should include delay spread and other parameters from CHANNEL_CONFIG
    return RayleighBlockFading(
        num_rx=num_users,
        num_rx_ant=MIMO_CONFIG["rx_antennas"],
        num_tx=1,
        num_tx_ant=MIMO_CONFIG["tx_antennas"],
        dtype=tf.complex64,
        delay_spread=CHANNEL_CONFIG["delay_spread"],  # Add this
        num_paths=CHANNEL_CONFIG["num_paths"],        # Add this
        path_loss_model=CHANNEL_CONFIG["path_loss_model"]  # Add this
    )

def generate_dataset(output_file, num_samples):
    """Generate dataset with improved validation and error checking"""
    print(f"Generating dataset: {output_file} with {num_samples} samples...")
    
    # Add validation checks
    if num_samples % SIONNA_CONFIG["batch_size"] != 0:
        raise ValueError("num_samples must be divisible by batch_size")
    
    # Initialize with proper shapes
    num_users = CONFIG.get("num_users", 4)
    batch_size = SIONNA_CONFIG["batch_size"]
    
    # Create channel model - Add this line
    channel_model = create_channel_model(num_users)
    
    dataset = {
        "channel_realizations": np.zeros((num_samples, MIMO_CONFIG["rx_antennas"], 
                                    num_users, 1, MIMO_CONFIG["tx_antennas"], 1, 1), 
                                    dtype=np.complex64),
        "snr": np.zeros(num_samples),
        "sinr": np.zeros((num_samples, num_users)),
        "interference": np.zeros((num_samples, num_users)),
        "user_association": np.zeros((num_samples, num_users)),
        "precoding_matrices": np.zeros((num_samples, MIMO_CONFIG["tx_antennas"], 
                                    MIMO_CONFIG["rx_antennas"]), dtype=np.complex64)
    }
    
    # Calculate noise power from noise floor
    noise_power = db2lin(CONFIG["noise_floor"])
    
    # Create base user association matrix (one-hot encoding)
    base_user_association = np.eye(num_users)
    
    # Generate data in batches
    for batch in range(num_samples // SIONNA_CONFIG["batch_size"]):
        batch_size = SIONNA_CONFIG["batch_size"]
        
        # Generate SNRs
        snrs = np.random.uniform(
            CHANNEL_CONFIG["snr_range"][0],
            CHANNEL_CONFIG["snr_range"][1],
            batch_size
        )
        
        # Generate channel realizations - Use channel_model instead of channel
        channels = channel_model(
            batch_size=batch_size,
            num_time_steps=1
        )
        if isinstance(channels, tuple):
            channels, _ = channels
        
        channels = channels.numpy()
        
        
        # Initialize arrays for SINR and interference calculations
        sinr_values = np.zeros((batch_size, num_users))
        interference_values = np.zeros((batch_size, num_users))
        
        # Reshape channels to combine relevant dimensions
        channels_reshaped = channels.reshape(
            batch_size, 
            num_users,
            -1
        )
        
        # Generate interference values directly in dB scale
        interference_values = np.random.uniform(
            CHANNEL_CONFIG["interference"]["interference_power_range"][0],
            CHANNEL_CONFIG["interference"]["interference_power_range"][1],
            (batch_size, num_users)
        )
        
        # Process each user
        for user in range(num_users):
            # Calculate signal power for the current user
            user_channel = channels_reshaped[:, user, :]
            signal_power = np.sum(np.abs(user_channel)**2, axis=1)
            
            # Apply antenna gain
            antenna_gain = MIMO_CONFIG["antenna_gain"]
            if isinstance(antenna_gain, tuple):
                antenna_gain = np.mean(antenna_gain)
            signal_power *= db2lin(antenna_gain)
            
            # Convert interference from dB to linear for SINR calculation
            interference_power_linear = db2lin(interference_values[:, user])
            
            # Calculate SINR
            sinr = signal_power / (interference_power_linear + noise_power)
            sinr_values[:, user] = lin2db(sinr)
        
        # Generate precoding matrices
        precoding = np.random.normal(
            0, 1/np.sqrt(2), 
            (batch_size, MIMO_CONFIG["tx_antennas"], MIMO_CONFIG["rx_antennas"])
        ) + 1j * np.random.normal(
            0, 1/np.sqrt(2), 
            (batch_size, MIMO_CONFIG["tx_antennas"], MIMO_CONFIG["rx_antennas"])
        )
        
        # Create proper user association for this batch
        user_association = np.tile(base_user_association, (batch_size, 1))
        
        # Append batch data to dataset
        dataset["channel_realizations"].append(channels)
        dataset["snr"].append(snrs)
        dataset["sinr"].append(sinr_values)
        dataset["interference"].append(interference_values)
        dataset["user_association"].append(user_association)
        dataset["precoding_matrices"].append(precoding)
        
        if (batch + 1) % 10 == 0:
            print(f"Processed batch {batch + 1}/{num_samples // SIONNA_CONFIG['batch_size']}")
    
    # Concatenate all batches
    for key in dataset:
        dataset[key] = np.concatenate(dataset[key], axis=0) if len(dataset[key]) > 0 else np.array([])
    
    # Save dataset
    np.save(output_file, dataset)
    print(f"\nDataset saved to {output_file}")
    print(f"Dataset statistics:")
    print(f"Channel realizations shape: {dataset['channel_realizations'].shape}")
    print(f"SNR shape: {dataset['snr'].shape}")
    print(f"SINR shape: {dataset['sinr'].shape}")
    print(f"Interference shape: {dataset['interference'].shape}")
    print(f"User association shape: {dataset['user_association'].shape}")
    print(f"Precoding matrices shape: {dataset['precoding_matrices'].shape}")
    
    # Verify interference range
    print(f"Interference range: [{np.min(dataset['interference']):.2f}, {np.max(dataset['interference']):.2f}] dB")
    
    return dataset

def validate_dataset(dataset):
    """Validate dataset dimensions and values"""
    expected_shapes = {
        "channel_realizations": (None, MIMO_CONFIG["rx_antennas"], CONFIG["num_users"], 
                            1, MIMO_CONFIG["tx_antennas"], 1, 1),
        "snr": (None,),
        "sinr": (None, CONFIG["num_users"]),
        "interference": (None, CONFIG["num_users"]),
        "user_association": (None, CONFIG["num_users"]),
        "precoding_matrices": (None, MIMO_CONFIG["tx_antennas"], MIMO_CONFIG["rx_antennas"])
    }
    
    # Check shapes
    for key, expected_shape in expected_shapes.items():
        actual_shape = dataset[key].shape
        if len(actual_shape) != len(expected_shape):
            raise ValueError(f"Invalid shape for {key}: expected {expected_shape}, got {actual_shape}")
        
        for i, (actual, expected) in enumerate(zip(actual_shape[1:], expected_shape[1:])):
            if expected is not None and actual != expected:
                raise ValueError(f"Invalid dimension {i} for {key}: expected {expected}, got {actual}")
    
    # Check value ranges
    assert np.all(dataset["snr"] >= CHANNEL_CONFIG["snr_range"][0])
    assert np.all(dataset["snr"] <= CHANNEL_CONFIG["snr_range"][1])
    assert np.all(dataset["sinr"] >= -20) and np.all(dataset["sinr"] <= 30)
    assert np.all(np.isfinite(dataset["channel_realizations"]))

if __name__ == "__main__":
    try:
        # Generate datasets with validation
        for dataset_type in ["training", "validation", "test"]:
            dataset = generate_dataset(
                OUTPUT_FILES[f"{dataset_type}_data"], 
                CONFIG[f"{dataset_type}_size"]
            )
            validate_dataset(dataset)
            print(f"{dataset_type} dataset generated and validated successfully")
    except Exception as e:
        print(f"Error generating datasets: {str(e)}")