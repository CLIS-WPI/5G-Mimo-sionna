import numpy as np

# Update the dataset path to the correct location
dataset_path = r"C:\Users\snatanzi\5G-Mimo-sionna\5G-Mimo-sionna\data\test\test_data.npy"
data = np.load(dataset_path, allow_pickle=True).item()

print("Dataset Structure:")
print("=" * 50)
print(f"Data type: {type(data)}")

print("\nSample Data (First 5 rows where applicable):")
print("=" * 50)

# Channel Realizations
if "channel_realizations" in data:
    print("\nChannel Realizations (First 5 samples):")
    print("-" * 50)
    channel_data = data["channel_realizations"][:5]  # Get first 5 rows
    print(f"Shape of each sample: {channel_data[0].shape}")
    for i in range(5):
        print(f"\nSample {i+1}:")
        print(channel_data[i][0][0][0][:5])  # Show first 5 elements of first dimension

# SNR Values
if "snr" in data:
    print("\nSNR Values (First 5 samples):")
    print("-" * 50)
    print(data["snr"][:5])

# SINR Values
if "sinr" in data:
    print("\nSINR Values (First 5 samples):")
    print("-" * 50)
    print(data["sinr"][:5])

# Interference Values
if "interference" in data:
    print("\nInterference Values (First 5 samples):")
    print("-" * 50)
    print(data["interference"][:5])

# User Association
if "user_association" in data:
    print("\nUser Association (First 5 samples):")
    print("-" * 50)
    print(data["user_association"][:5])

# Precoding Matrices
if "precoding_matrices" in data:
    print("\nPrecoding Matrices (First 5 samples):")
    print("-" * 50)
    print(data["precoding_matrices"][:5])

# Doppler Information
if "doppler_info" in data:
    print("\nDoppler Information:")
    print("-" * 50)
    print(data["doppler_info"])

print("\nArray Shapes:")
print("=" * 50)
for key in data.keys():
    if isinstance(data[key], np.ndarray):
        print(f"{key}: {data[key].shape}")