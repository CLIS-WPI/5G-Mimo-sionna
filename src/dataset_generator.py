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
        num_rx=MIMO_CONFIG["rx_antennas"],
        num_tx=MIMO_CONFIG["tx_antennas"],
        num_paths=CHANNEL_CONFIG["num_paths"],
        coherence_time=CHANNEL_CONFIG["coherence_time"],
        delay_spread=CHANNEL_CONFIG["delay_spread"],
        doppler_frequency=CHANNEL_CONFIG["doppler_shift"],
        carrier_frequency=RESOURCE_GRID["bandwidth"],
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
        channels = channel(batch_size=SIONNA_CONFIG["batch_size"], tx_array=tx_array, rx_array=rx_array)

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
