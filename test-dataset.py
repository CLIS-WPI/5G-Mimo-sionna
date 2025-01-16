import numpy as np

# File paths
TRAINING_DATA_PATH = "./data/training/training_data.npy"
VALIDATION_DATA_PATH = "./data/validation/validation_data.npy"
TEST_DATA_PATH = "./data/test/test_data.npy"

# Load datasets
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    data = np.load(file_path, allow_pickle=True).item()
    return data

def check_snr_range(dataset, dataset_name, snr_range=(0, 30)):
    snr_values = dataset["snr"]
    min_snr = snr_values.min()
    max_snr = snr_values.max()
    print(f"  {dataset_name} SNR Range: {min_snr:.2f} to {max_snr:.2f}")
    if min_snr < snr_range[0] or max_snr > snr_range[1]:
        print(f"  Error: {dataset_name} contains out-of-range SNR values.")
        return False
    print(f"  {dataset_name} SNR values are within the valid range.")
    return True

def check_dataset(dataset, dataset_name):
    print(f"\nChecking {dataset_name} Dataset:")

    # Check keys
    if "channel_realizations" not in dataset or "snr" not in dataset:
        print(f"Error: Missing keys in {dataset_name} dataset.")
        return False

    # Check shapes
    channel_realizations = dataset["channel_realizations"]
    snr = dataset["snr"]

    print(f"  Channel Realizations Shape: {channel_realizations.shape}")
    print(f"  SNR Shape: {snr.shape}")

    # Ensure number of samples match
    if channel_realizations.shape[0] != snr.shape[0]:
        print(f"Error: Mismatch in number of samples between channel_realizations and snr in {dataset_name} dataset.")
        return False

    # Check MIMO dimensions
    num_rx_antennas = channel_realizations.shape[1]
    num_tx_antennas = channel_realizations.shape[2]
    print(f"  Number of RX Antennas: {num_rx_antennas}")
    print(f"  Number of TX Antennas: {num_tx_antennas}")

    # Print a small sample
    print("  Sample Channel Realization:")
    print(channel_realizations[0])
    print("  Sample SNR Value:")
    print(snr[0])

    # Check SNR range
    return check_snr_range(dataset, dataset_name)

if __name__ == "__main__":
    # Load datasets
    training_data = load_dataset(TRAINING_DATA_PATH)
    validation_data = load_dataset(VALIDATION_DATA_PATH)
    test_data = load_dataset(TEST_DATA_PATH)

    # Check datasets
    training_valid = check_dataset(training_data, "Training")
    validation_valid = check_dataset(validation_data, "Validation")
    test_valid = check_dataset(test_data, "Test")

    # Final summary
    if training_valid and validation_valid and test_valid:
        print("\nAll datasets are valid!")
    else:
        print("\nSome datasets are invalid. Please review the errors above.")
