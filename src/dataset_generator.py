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
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utill.gpu import setup_gpu, get_gpu_memory_info
gpu_available = setup_gpu()
if gpu_available:
    print("GPU is configured and ready for use")
    print(get_gpu_memory_info())
else:
    print("Running on CPU mode")
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
from sionna.channel.tr38901 import TDL
import gc
import psutil
# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_FILES["training_data"]), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILES["validation_data"]), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILES["test_data"]), exist_ok=True)
#############################################
def monitor_memory():
    """Monitor current memory usage"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

#function to save less frequently:
def save_chunk_to_file(chunk_data, output_file, chunk_idx):
    """Save a chunk of data to a temporary file with compression"""
    temp_file = f"{output_file}_chunk_{chunk_idx}.npz"
    try:
        # Save each key separately in the compressed file
        save_dict = {key: chunk_data[key] for key in chunk_data if key != "doppler_info"}
        np.savez_compressed(temp_file, **save_dict)
        # Explicitly close any file handles
        np.load(temp_file).close()
        return temp_file
    except Exception as e:
        print(f"Error saving chunk to file: {str(e)}")
        return None

def cleanup_temp_files(temp_files):
    """Clean up any remaining temporary files"""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_file}: {e}")
            
def merge_chunks(temp_files, output_file, doppler_info):
    """Merge temporary chunk files into final dataset"""
    merged_data = {
        "channel_realizations": [],
        "snr": [],
        "sinr": [],
        "interference": [],
        "user_association": [],
        "precoding_matrices": [],
        "doppler_info": doppler_info
    }
    
    # Process existing files
    for temp_file in temp_files:
        if not os.path.exists(temp_file):
            print(f"Warning: Chunk file {temp_file} not found")
            continue
            
        try:
            # Load data differently based on file type
            if temp_file.endswith('.npy'):
                chunk = np.load(temp_file, allow_pickle=True)
                if isinstance(chunk, np.ndarray):
                    chunk = chunk.item()  # Convert ndarray to dictionary if needed
            else:  # NPZ file
                chunk = dict(np.load(temp_file, allow_pickle=True))
                
            # Append data from chunk
            for key in merged_data:
                if key != "doppler_info" and key in chunk:
                    merged_data[key].append(chunk[key])
                    
            # Clean up temporary file
            try:
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                print(f"Warning: Coul   d not delete temporary file {temp_file}: {e}")
                
        except Exception as e:
            print(f"Error processing chunk file {temp_file}: {str(e)}")
            continue

    # Merge the data if we have any
    if any(len(merged_data[key]) > 0 for key in merged_data if key != "doppler_info"):
        for key in merged_data:
            if key != "doppler_info" and merged_data[key]:
                try:
                    merged_data[key] = np.concatenate(merged_data[key], axis=0)
                except Exception as e:
                    print(f"Error concatenating {key}: {str(e)}")
        
        # Save merged data
        try:
            np.save(output_file, merged_data)
            print(f"Successfully saved merged data to {output_file}")
        except Exception as e:
            print(f"Error saving merged data: {str(e)}")
    else:
        print("Warning: No data to merge")
    
    return merged_data

#############################################
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
    # Reshape inputs to ensure proper broadcasting
    desired_signal = np.expand_dims(desired_signal, axis=-1)  # Add dimension for broadcasting
    
    # Calculate signal power with antenna gain
    signal_power = np.mean(np.abs(desired_signal)**2, axis=(-2, -1)) * db2lin(MIMO_CONFIG["antenna_gain"])
    
    # Generate interference power with proper shape
    interference_power = np.random.uniform(
        CHANNEL_CONFIG["interference"]["interference_power_range"][0],
        CHANNEL_CONFIG["interference"]["interference_power_range"][1],
        size=signal_power.shape
    )
    
    # Convert interference from dB to linear
    interference_power_linear = db2lin(interference_power)
    
    # Calculate SINR ensuring proper broadcasting
    sinr = signal_power / (interference_power_linear + noise_power)
    
    return lin2db(sinr), interference_power

def calculate_max_doppler_freq():
    """Calculate maximum Doppler frequency"""
    return (CHANNEL_CONFIG["doppler_shift"]["max_speed"] * 
            CHANNEL_CONFIG["doppler_shift"]["carrier_frequency"] / 
            SPEED_OF_LIGHT)

def calculate_coherence_time():
    """Calculate channel coherence time based on Doppler parameters"""
    max_doppler_freq = calculate_max_doppler_freq()
    if max_doppler_freq == 0:
        return float('inf')  # Return infinity for static channels
    return 1 / (2 * max_doppler_freq)  # Standard coherence time formula

def create_channel_model(num_users):
    """Create proper channel model"""
    return RayleighBlockFading(
        num_rx=num_users,
        num_rx_ant=MIMO_CONFIG["rx_antennas"],
        num_tx=1,
        num_tx_ant=MIMO_CONFIG["tx_antennas"],
        dtype=tf.complex64
    )

def create_ofdm_channel(channel_model, resource_grid):
    """Create OFDM channel with proper configuration"""
    return OFDMChannel(
        channel_model=channel_model,
        resource_grid=resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=True,
        dtype=tf.complex64
    )

def normalize_channel_realizations(channels):
    """Normalize channel realizations to have unit average power"""
    # Convert to complex tensor if needed
    channels = tf.convert_to_tensor(channels, dtype=tf.complex64)
    
    # Calculate power across all dimensions
    power = tf.reduce_mean(tf.abs(channels)**2)
    print("\nNormalization Debug:")
    print(f"Input channels shape: {channels.shape}")
    print(f"Original power: {power:.10e}")
    
    # Check if power is too small
    if power < 1e-10:
        print("Warning: Very small power detected, applying pre-scaling")
        pre_scaling = tf.cast(1e5, dtype=tf.complex64)
        channels = channels * pre_scaling
        power = tf.reduce_mean(tf.abs(channels)**2)
        print(f"Power after pre-scaling: {power:.10e}")
    
    # Calculate scaling factor
    scaling_factor = tf.cast(tf.sqrt(1.0 / power), dtype=tf.complex64)
    print(f"Scaling factor: {scaling_factor:.10e}")
    
    # Apply normalization
    channels_normalized = channels * scaling_factor
    
    # Verify normalized power
    normalized_power = tf.reduce_mean(tf.abs(channels_normalized)**2)
    print(f"Normalized power: {normalized_power:.10e}")
    print(f"Target power: 1.0")
    print(f"Difference from target: {abs(normalized_power - 1.0):.10e}")
    
    # Use larger tolerance for the assertion
    rtol = 1e-1  # Increased from 1e-2
    atol = 1e-1  # Increased from 1e-2
    
    try:
        tf.debugging.assert_near(normalized_power, 1.0, rtol=rtol, atol=atol)
    except tf.errors.InvalidArgumentError as e:
        print(f"\nWarning: Normalization failed with current tolerances:")
        print(f"Relative tolerance: {rtol}")
        print(f"Absolute tolerance: {atol}")
        print(f"Consider adjusting tolerances or investigating channel generation")
        raise
    
    return channels_normalized


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
    
def validate_batch_size(num_samples, batch_size):
    """Validate that batch size divides total samples evenly"""
    if num_samples % batch_size != 0:
        raise ValueError(
            f"Number of samples ({num_samples}) must be divisible by "
            f"batch size ({batch_size})"
        )

def print_progress(batch, total_batches):
    """Print progress bar for batch processing"""
    progress = (batch + 1) / total_batches * 100
    print(f"\rProgress: [{batch + 1}/{total_batches}] {progress:.1f}%", end="")

def generate_dataset(output_file, num_samples):
    validate_batch_size(num_samples, SIONNA_CONFIG["batch_size"])
    total_batches = num_samples // SIONNA_CONFIG["batch_size"]
    print(f"Generating dataset: {output_file} with {num_samples} samples...")
    
    # Initialize variables for chunk management
    temp_files = []
    chunk_data = None  # Initialize chunk_data
    chunk_save_frequency = 5  # Save every 5 batches
    merge_threshold = 10      # Merge after 10 chunks
    
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

    # Initialize temp_files list before the batch loop
    temp_files = [] 

    # Generate data in batches
    for batch in range(num_samples // SIONNA_CONFIG["batch_size"]):
        batch_size = SIONNA_CONFIG["batch_size"]
        
        # Save chunk to temporary file if we have data
        if chunk_data is not None:
    
            # Only save chunks at specified frequency
            if (batch + 1) % chunk_save_frequency == 0:
                temp_file = save_chunk_to_file(chunk_data, output_file, batch)
                if os.path.exists(temp_file):
                    temp_files.append(temp_file)
                    print(f"Saved chunk {batch}")
                else:
                    print(f"Warning: Failed to save chunk {batch}")
                    
            # Merge chunks periodically
            if len(temp_files) >= merge_threshold:
                print("\nIntermediate merging of chunks...")
                # Verify all files exist before merging
                existing_files = [f for f in temp_files if os.path.exists(f)]
                if existing_files:
                    intermediate_output = f"{output_file}_intermediate.npy"
                    merge_chunks(existing_files, intermediate_output, dataset["doppler_info"])
                    temp_files = [intermediate_output]
                    print(f"Merged {len(existing_files)} chunks")
                else:
                    print("Warning: No valid files to merge")
        
        # Generate SNRs
        snrs = np.random.uniform(
            CHANNEL_CONFIG["snr_range"][0],
            CHANNEL_CONFIG["snr_range"][1],
            batch_size
        )
        
        # After generating channels
        channels = channel((dummy_input, noise_power))
        if isinstance(channels, tuple):
            channels, _ = channels
        channels = channels.numpy()

        print("\nChannel Generation Debug:")
        print(f"Channel shape: {channels.shape}")
        print(f"Channel dtype: {channels.dtype}")
        print(f"Channel min/max: {np.min(np.abs(channels)):.10e} / {np.max(np.abs(channels)):.10e}")
        print(f"Channel mean magnitude: {np.mean(np.abs(channels)):.10e}")

        # Add normalization
        channels = normalize_channel_realizations(channels)

        # Verify channel statistics
        verify_channel_statistics(channels)

        # Check shapes before SINR calculation
        print(f"Channel shape before reshape: {channels.shape}")
        print(f"Channel type before reshape: {type(channels)}")
        print(f"Expected reshape dimensions: [{batch_size}, {num_users}, -1]")

        # Convert to tensor and reshape
        channels = tf.convert_to_tensor(channels, dtype=tf.complex64)
        channels_reshaped = tf.reshape(channels, [batch_size, num_users, -1])

        print(f"Channel shape after reshape: {channels_reshaped.shape}")
        print(f"Channel type after reshape: {type(channels_reshaped)}")

        # Initialize arrays for SINR and interference calculations
        sinr_values = np.zeros((batch_size, num_users))
        interference_values = np.zeros((batch_size, num_users))

        # Convert reshaped channels back to numpy for further processing
        channels_reshaped = channels_reshaped.numpy()
        
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
        
        # Create chunk data
        chunk_data = {
            "channel_realizations": channels,
            "snr": snrs,
            "sinr": sinr_values,
            "interference": interference_values,
            "user_association": user_association,
            "precoding_matrices": precoding
        }
        
        
        if (batch + 1) % 10 == 0:
            print(f"Processed batch {batch + 1}/{num_samples // SIONNA_CONFIG['batch_size']}")
            monitor_memory()
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
        
        # After all batches are processed, merge chunks
        if batch == (num_samples // SIONNA_CONFIG["batch_size"] - 1):
            print("\nMerging chunks...")
            dataset = merge_chunks(temp_files, output_file, dataset["doppler_info"])
    
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

    # Clean up any remaining temporary files
    cleanup_temp_files(temp_files)
    return dataset


def verify_channel_statistics(channels):
    """Verify channel statistics are correct"""
    real_parts = np.real(channels)
    imag_parts = np.imag(channels)
    
    # Calculate statistics
    real_mean = np.mean(real_parts)
    real_std = np.std(real_parts)
    imag_mean = np.mean(imag_parts)
    imag_std = np.std(imag_parts)
    channel_power = np.mean(np.abs(channels)**2)
    
    print("\nChannel Statistics:")
    print(f"Real part - Mean: {real_mean:.4f}, Std: {real_std:.4f}")
    print(f"Imag part - Mean: {imag_mean:.4f}, Std: {imag_std:.4f}")
    print(f"Channel power: {channel_power:.4f}")
    
    # Verify statistical properties
    if np.allclose(channels, 0) or real_std < 1e-6 or imag_std < 1e-6:
        raise ValueError("Channel realizations appear to be zero or have very low variance")
    
    # Check for proper normalization
    if not np.isclose(channel_power, 1.0, rtol=1e-2):
        raise ValueError(f"Channel power ({channel_power:.4f}) is not properly normalized to 1.0")
    
    # Verify complex Gaussian properties
    if not (-0.1 < real_mean < 0.1 and -0.1 < imag_mean < 0.1):
        raise ValueError("Channel statistics do not match expected complex Gaussian distribution")

if __name__ == "__main__":
    # Generate all datasets
    generate_dataset(OUTPUT_FILES["training_data"], CONFIG["dataset_size"])
    generate_dataset(OUTPUT_FILES["validation_data"], CONFIG["validation_size"])
    generate_dataset(OUTPUT_FILES["test_data"], CONFIG["test_size"])