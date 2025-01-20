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
from sionna import SPEED_OF_LIGHT
from sionna.channel import OFDMChannel, RayleighBlockFading
from sionna.channel.tr38901 import AntennaArray
from sionna.mimo import StreamManagement
from sionna.mimo.precoding import zero_forcing_precoder, normalize_precoding_power
from utill.utils import db2lin, lin2db
from config import CONFIG, MIMO_CONFIG, RESOURCE_GRID, CHANNEL_CONFIG, SIONNA_CONFIG, OUTPUT_FILES
from sionna.ofdm import ResourceGrid
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

rg = ResourceGrid(
    # Required parameters
    num_ofdm_symbols=RESOURCE_GRID["ofdm_symbols"],     # Must be specified
    fft_size=RESOURCE_GRID["subcarriers"],              # FFT size
    subcarrier_spacing=RESOURCE_GRID["subcarrier_spacing"], 
    
    # Optional parameters (with your config values)
    num_guard_carriers=RESOURCE_GRID["num_guard_carriers"],  # Tuple of (left, right) guards
    dc_null=RESOURCE_GRID["dc_null"],                       # Boolean for DC nulling
    
    # Other optional parameters with default values
    num_tx=1,                     # Number of transmitters
    num_streams_per_tx=1,         # Streams per transmitter
    cyclic_prefix_length=0,       # CP length
    pilot_pattern="empty"         # Pilot pattern configuration
)

def create_stream_management(num_users):
    """Create stream management for multi-user scenario"""
    rx_tx_association = np.zeros([num_users, MIMO_CONFIG["tx_antennas"]])
    # Assign each transmitter to a receiver (round-robin)
    for i in range(MIMO_CONFIG["tx_antennas"]):
        rx_tx_association[i % num_users, i] = 1
    
    return StreamManagement(rx_tx_association, num_streams_per_tx=1)

def calculate_sinr(desired_signal, interference_signals, noise_power):
    # Calculate signal power with antenna gain
    signal_power = np.abs(desired_signal)**2 * db2lin(MIMO_CONFIG["antenna_gain"])
    
    # Calculate interference power directly in dB scale
    interference_power = np.random.uniform(
        CHANNEL_CONFIG["interference"]["interference_power_range"][0],
        CHANNEL_CONFIG["interference"]["interference_power_range"][1],
        size=signal_power.shape
    )
    
    # Convert interference from dB to linear for SINR calculation
    interference_power_linear = db2lin(interference_power)
    
    # Calculate SINR
    sinr = signal_power / (interference_power_linear + noise_power)
    
    return lin2db(sinr), interference_power  # Return both SINR and interference in dB

def calculate_max_doppler_freq():
    """Calculate maximum Doppler frequency"""
    return (CHANNEL_CONFIG["doppler_shift"]["max_speed"] * 
            CHANNEL_CONFIG["doppler_shift"]["carrier_frequency"] / 
            SPEED_OF_LIGHT)

def create_channel_model(num_users):
    """Create channel model with proper Doppler configuration"""
    
    return RayleighBlockFading(
        num_rx=num_users,
        num_rx_ant=MIMO_CONFIG["rx_antennas"],
        num_tx=1,
        num_tx_ant=MIMO_CONFIG["tx_antennas"],
        dtype=tf.complex64
    )

def calculate_coherence_time():
    """Calculate channel coherence time based on Doppler parameters"""
    max_doppler = 2 * np.pi * CHANNEL_CONFIG["doppler_shift"]["max_speed"] / \
                 SPEED_OF_LIGHT * CHANNEL_CONFIG["doppler_shift"]["carrier_frequency"]
    return 1 / max_doppler


def create_resource_grid():
    """Create resource grid with proper configuration"""
    return ResourceGrid(
        num_ofdm_symbols=RESOURCE_GRID["ofdm_symbols"],
        fft_size=RESOURCE_GRID["subcarriers"],
        subcarrier_spacing=RESOURCE_GRID["subcarrier_spacing"],
        num_guard_carriers=RESOURCE_GRID["num_guard_carriers"],
        dc_null=RESOURCE_GRID["dc_null"],
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=0,
        pilot_pattern="empty"
    )

def validate_doppler_params():
    """Validate Doppler shift parameters"""
    try:
        doppler_params = CHANNEL_CONFIG["doppler_shift"]
        assert doppler_params["min_speed"] >= 0, "Minimum speed must be non-negative"  # Changed from > 0 to >= 0
        assert doppler_params["max_speed"] > doppler_params["min_speed"], \
            "Maximum speed must be greater than minimum speed"
        assert doppler_params["carrier_frequency"] > 0, "Carrier frequency must be positive"
    except AssertionError as e:
        raise ValueError(f"Invalid Doppler parameters: {str(e)}")
    except KeyError as e:
        raise KeyError(f"Missing Doppler parameter: {str(e)}")

def generate_dataset(output_file, num_samples):
    """Generate dataset with improved SINR calculation and multi-user support"""
    print(f"Generating dataset: {output_file} with {num_samples} samples...")
    
    # Calculate and print coherence time at the start
    coherence_time = calculate_coherence_time()
    print(f"Channel coherence time: {coherence_time*1000:.2f} ms")

    # Set number of users
    num_users = CONFIG.get("num_users", 4)

    # Create resource grid
    resource_grid = create_resource_grid()

    # Validate Doppler parameters
    validate_doppler_params()

    # Create channel model with OFDM parameters
    channel = OFDMChannel(
        channel_model=create_channel_model(num_users),
        resource_grid=resource_grid,
        add_awgn=True,
        normalize_channel=True,
        dtype=tf.complex64
    )
    
    # Update dataset dictionary to include Doppler information
    dataset = {
        "channel_realizations": [],
        "snr": [],
        "sinr": [],
        "interference": [],
        "user_association": [],
        "precoding_matrices": [],
        "doppler_info": {  # Add Doppler information
            "max_speed": CHANNEL_CONFIG["doppler_shift"]["max_speed"],
            "min_speed": CHANNEL_CONFIG["doppler_shift"]["min_speed"],
            "carrier_frequency": CHANNEL_CONFIG["doppler_shift"]["carrier_frequency"]
        }
    }

    # Calculate noise power from noise floor
    noise_power = db2lin(CONFIG["noise_floor"])
    
    # Create base user association matrix (one-hot encoding)
    base_user_association = np.eye(num_users)
    
    # Create dummy input tensor for channel model
    dummy_input = tf.zeros([
        SIONNA_CONFIG["batch_size"],  # batch size
        1,                            # num_tx
        MIMO_CONFIG["tx_antennas"],   # num_tx_ant
        RESOURCE_GRID["ofdm_symbols"],# num_ofdm_symbols
        RESOURCE_GRID["subcarriers"]  # fft_size
    ], dtype=tf.complex64)

    # Generate data in batches
    for batch in range(num_samples // SIONNA_CONFIG["batch_size"]):
        batch_size = SIONNA_CONFIG["batch_size"]
        
        # Generate SNRs
        snrs = np.random.uniform(
            CHANNEL_CONFIG["snr_range"][0],
            CHANNEL_CONFIG["snr_range"][1],
            batch_size
        )
        
        # Generate channel realizations
        channels = channel((dummy_input, noise_power))
        
        # Handle the case where channels is a tuple
        if isinstance(channels, tuple):
            channels, _ = channels
        
        # Convert to numpy array for storage
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
    print(f"\nDataset Statistics:")
    print("=" * 50)
    
    # Print shapes and basic statistics
    print("\nShape Information:")
    print(f"Channel realizations: {dataset['channel_realizations'].shape}")
    print(f"SNR values: {dataset['snr'].shape}")
    print(f"SINR values: {dataset['sinr'].shape}")
    print(f"Interference values: {dataset['interference'].shape}")
    print(f"User association: {dataset['user_association'].shape}")
    print(f"Precoding matrices: {dataset['precoding_matrices'].shape}")
    
    print("\nValue Ranges:")
    print(f"SNR range: [{np.min(dataset['snr']):.2f}, {np.max(dataset['snr']):.2f}] dB")
    print(f"SINR range: [{np.min(dataset['sinr']):.2f}, {np.max(dataset['sinr']):.2f}] dB")
    print(f"Interference range: [{np.min(dataset['interference']):.2f}, {np.max(dataset['interference']):.2f}] dB")
    
    print("\nDoppler Information:")
    print(f"Carrier Frequency: {dataset['doppler_info']['carrier_frequency']/1e9:.2f} GHz")
    print(f"Speed Range: [{dataset['doppler_info']['min_speed']:.1f}, "
        f"{dataset['doppler_info']['max_speed']:.1f}] m/s")
    print(f"Coherence Time: {calculate_coherence_time()*1000:.2f} ms")
    
    # Verify data integrity
    print("\nData Integrity Checks:")
    print(f"Number of samples: {len(dataset['channel_realizations'])}")
    print(f"Memory usage: {sum([x.nbytes for x in dataset.values() if isinstance(x, np.ndarray)])/1e6:.2f} MB")
    
    # Check for NaN or Inf values
    has_nan = any(np.isnan(x).any() for x in dataset.values() if isinstance(x, np.ndarray))
    has_inf = any(np.isinf(x).any() for x in dataset.values() if isinstance(x, np.ndarray))
    print(f"Contains NaN values: {'Yes' if has_nan else 'No'}")
    print(f"Contains Inf values: {'Yes' if has_inf else 'No'}")
    
    print("\n" + "=" * 50)
    
    return dataset

if __name__ == "__main__":
    # Generate all datasets
    generate_dataset(OUTPUT_FILES["training_data"], CONFIG["dataset_size"])
    generate_dataset(OUTPUT_FILES["validation_data"], CONFIG["validation_size"])
    generate_dataset(OUTPUT_FILES["test_data"], CONFIG["test_size"])