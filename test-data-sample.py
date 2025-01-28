import numpy as np

# Update the dataset path to the correct location
dataset_path = r"C:\Users\snatanzi\5G-Mimo-sionna\5G-Mimo-sionna\data\test\test_data.npy"
data = np.load(dataset_path, allow_pickle=True)

# Print dataset information
print("Dataset Structure:")
print("=" * 50)
print(f"Data type: {type(data)}")
print(f"Data shape: {data.shape}")

# Since we're working with test configuration files, let's print their contents
print("\nData Contents:")
print("=" * 50)
for i, item in enumerate(data):
    print(f"\nItem {i}:")
    print(f"Type: {type(item)}")
    if hasattr(item, 'shape'):
        print(f"Shape: {item.shape}")
    print(f"Content: {item}")

# If the data structure is different from what we expected, we can examine it
print("\nDetailed Examination:")
print("=" * 50)
try:
    # Try to access some common attributes or methods
    if hasattr(data, 'keys'):
        print("Dictionary keys:", data.keys())
    elif isinstance(data, np.ndarray):
        print("Array attributes:")
        print(f"dtype: {data.dtype}")
        print(f"size: {data.size}")
        print(f"ndim: {data.ndim}")
except Exception as e:
    print(f"Error examining data: {e}")