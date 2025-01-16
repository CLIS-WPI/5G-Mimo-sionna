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
import numpy as np
import tensorflow as tf
from sionna.channel import OFDMChannel, RayleighBlockFading
from sionna.channel.tr38901 import AntennaArray
from config import CONFIG, MIMO_CONFIG, RESOURCE_GRID, CHANNEL_CONFIG, SIONNA_CONFIG, OUTPUT_FILES

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_FILES["training_data"]), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILES["validation_data"]), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILES["test_data"]), exist_ok=True)

# Define the Antenna Array
def create_antenna_array():
    tx_array = AntennaArray(
        num_rows=1,
        num_cols=MIMO_CONFIG["tx_antennas"],
        polarization=MIMO_CONFIG["polarization"],
        polarization_type="V",  # Add this parameter
        antenna_pattern="38.901",  # Add this parameter
        carrier_frequency=RESOURCE_GRID["bandwidth"],
        vertical_spacing=MIMO_CONFIG["element_spacing"],    # Changed from element_spacing
        horizontal_spacing=MIMO_CONFIG["element_spacing"],  # Changed from element_spacing
    )
    rx_array = AntennaArray(
        num_rows=1,
        num_cols=MIMO_CONFIG["rx_antennas"],
        polarization=MIMO_CONFIG["polarization"],
        polarization_type="V",  # Add this parameter
        antenna_pattern="38.901",  # Add this parameter
        carrier_frequency=RESOURCE_GRID["bandwidth"],
        vertical_spacing=MIMO_CONFIG["element_spacing"],    # Changed from element_spacing
        horizontal_spacing=MIMO_CONFIG["element_spacing"],  # Changed from element_spacing
    )
    return tx_array, rx_array

# Define the channel model
def create_channel_model():
    return RayleighBlockFading(
        num_rx=1,  # Since we're using num_rx_ant for the actual antenna count
        num_rx_ant=MIMO_CONFIG["rx_antennas"],
        num_tx=1,  # Since we're using num_tx_ant for the actual antenna count
        num_tx_ant=MIMO_CONFIG["tx_antennas"],
        dtype=tf.complex64
    )

# Generate the dataset
def generate_dataset(output_file, num_samples):
    print(f"Generating dataset: {output_file} with {num_samples} samples...")

    # Create the antenna arrays
    tx_array, rx_array = create_antenna_array()

    # Create the channel model
    channel = create_channel_model()

    # Initialize arrays for dataset
    dataset = {
        "channel_realizations": [],
        "snr": [],
    }

    for _ in range(num_samples // SIONNA_CONFIG["batch_size"]):
        # Generate random SNRs
        snrs = np.random.uniform(CHANNEL_CONFIG["snr_range"][0], CHANNEL_CONFIG["snr_range"][1], SIONNA_CONFIG["batch_size"])

        # Generate channel realizations
        # Modified this line to use correct parameters
        channels, _ = channel(
            batch_size=SIONNA_CONFIG["batch_size"],
            num_time_steps=1  # For block fading, we can use 1 time step
        )

        # Append to dataset
        dataset["channel_realizations"].append(channels.numpy())
        dataset["snr"].append(snrs)

    # Save the dataset to the specified output file
    dataset["channel_realizations"] = np.concatenate(dataset["channel_realizations"], axis=0)
    dataset["snr"] = np.concatenate(dataset["snr"], axis=0)
    np.save(output_file, dataset)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    # Generate training dataset
    generate_dataset(OUTPUT_FILES["training_data"], CONFIG["dataset_size"])

    # Generate validation dataset
    generate_dataset(OUTPUT_FILES["validation_data"], CONFIG["validation_size"])

    # Generate test dataset
    generate_dataset(OUTPUT_FILES["test_data"], CONFIG["test_size"])
