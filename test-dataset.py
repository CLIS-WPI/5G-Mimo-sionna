import numpy as np
from src.config import CONFIG, MIMO_CONFIG, CHANNEL_CONFIG, MULTIUSER_CONFIG

# File paths
TRAINING_DATA_PATH = "./data/training/training_data.npy"
VALIDATION_DATA_PATH = "./data/validation/validation_data.npy"
TEST_DATA_PATH = "./data/test/test_data.npy"

def load_dataset(file_path):
    """Load dataset with error handling"""
    try:
        print(f"Loading dataset from {file_path}...")
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {str(e)}")
        return None

def check_required_keys(dataset, dataset_name):
    """Verify all required keys are present in the dataset"""
    required_keys = [
        "channel_realizations", 
        "snr", 
        "sinr", 
        "interference",
        "user_association",
        "precoding_matrices"
    ]
    
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        print(f"Error: Missing keys in {dataset_name} dataset: {missing_keys}")
        return False
    return True

def check_dimensions(dataset, dataset_name):
    """Verify dimensions of all dataset components"""
    num_samples = dataset["channel_realizations"].shape[0]
    num_users = MULTIUSER_CONFIG["num_users"]
    
    checks = {
        "channel_realizations": (num_samples, num_users, MIMO_CONFIG["rx_antennas"], 
                            1, MIMO_CONFIG["tx_antennas"], 1, 1),
        "snr": (num_samples,),
        "sinr": (num_samples, num_users),
        "interference": (num_samples, num_users),
        "precoding_matrices": (num_samples, MIMO_CONFIG["tx_antennas"], MIMO_CONFIG["rx_antennas"])
    }
    
    for key, expected_shape in checks.items():
        actual_shape = dataset[key].shape
        if actual_shape != expected_shape:
            print(f"Error: Shape mismatch in {dataset_name} - {key}")
            print(f"Expected shape: {expected_shape}, Got: {actual_shape}")
            return False
    return True

def check_value_ranges(dataset, dataset_name):
    """Verify value ranges for different metrics"""
    checks = {
        "snr": {
            "range": CHANNEL_CONFIG["snr_range"],
            "values": dataset["snr"]
        },
        "sinr": {
            "min": -20,  # Reasonable minimum SINR
            "max": max(CHANNEL_CONFIG["snr_range"][1], CONFIG["sinr_target"]),
            "values": dataset["sinr"]
        },
        "interference": {
            "min": CHANNEL_CONFIG["interference"]["interference_power_range"][0],
            "max": CHANNEL_CONFIG["interference"]["interference_power_range"][1],
            "values": dataset["interference"]
        }
    }
    
    valid = True
    for metric, check in checks.items():
        values = check["values"]
        if "range" in check:
            min_val, max_val = check["range"]
        else:
            min_val = check["min"]
            max_val = check["max"]
            
        actual_min = np.min(values)
        actual_max = np.max(values)
        
        print(f"  {dataset_name} {metric} range: {actual_min:.2f} to {actual_max:.2f}")
        
        # Check if values are within expected range
        if actual_min < min_val or actual_max > max_val:
            print(f"  Warning: {metric} values outside expected range [{min_val}, {max_val}]")
            valid = False
            
        # Additional checks for specific metrics
        if metric == "interference" and np.any(values < 0):
            print(f"  Error: Negative interference values found in {dataset_name}")
            valid = False
        elif metric == "sinr":
            # SINR should typically be lower than SNR
            if np.any(values > dataset["snr"].max() + 10):  # Allow some margin
                print(f"  Warning: Some SINR values in {dataset_name} exceed SNR by more than 10dB")
                valid = False
    
    return valid

def check_complex_values(dataset, dataset_name):
    """Verify complex-valued data"""
    # Check channel realizations
    if not np.iscomplexobj(dataset["channel_realizations"]):
        print(f"Error: Channel realizations in {dataset_name} should be complex-valued")
        return False
    
    # Check precoding matrices
    if not np.iscomplexobj(dataset["precoding_matrices"]):
        print(f"Error: Precoding matrices in {dataset_name} should be complex-valued")
        return False
    
    return True

def check_sinr_range(dataset, dataset_name, sinr_range=(-20, 30)):
    sinr_values = dataset["sinr"]
    min_sinr = sinr_values.min()
    max_sinr = sinr_values.max()
    print(f"  {dataset_name} SINR Range: {min_sinr:.2f} to {max_sinr:.2f}")
    if min_sinr < sinr_range[0] or max_sinr > sinr_range[1]:
        print(f"  Warning: SINR values outside expected range {sinr_range}")
        return False
    return True

def check_interference_range(dataset, dataset_name):
    interference_values = dataset["interference"]
    min_int = interference_values.min()
    max_int = interference_values.max()
    print(f"  {dataset_name} Interference Range: {min_int:.2f} to {max_int:.2f}")
    # Check if interference is within the configured range (-100, -80)
    if min_int < -100 or max_int > -80:
        print(f"  Error: Interference values outside configured range [-100, -80]")
        return False
    return True

def check_user_association_dimensions(dataset, dataset_name):
    user_association = dataset["user_association"]
    batch_size = dataset["channel_realizations"].shape[0]
    num_users = CONFIG["num_users"]
    expected_shape = (batch_size, num_users)
    
    if user_association.shape != expected_shape:
        print(f"  Error: User association has shape {user_association.shape}, "
            f"expected {expected_shape}")
        return False
    return True

def check_snr_range(dataset, dataset_name, snr_range):
    """Verify SNR values are within the expected range"""
    snr_values = dataset["snr"]
    min_snr = snr_values.min()
    max_snr = snr_values.max()
    print(f"  {dataset_name} SNR range: {min_snr:.2f} to {max_snr:.2f}")
    
    if min_snr < snr_range[0] or max_snr > snr_range[1]:
        print(f"  Warning: SNR values outside expected range {snr_range}")
        return False
    return True

def print_validation_summary(validation_details):
    # Add this check before printing "SUCCESS"
    all_valid = all(details["valid"] for details in validation_details.values())
    
    # Only print success if all datasets are actually valid
    if all_valid:
        print("SUCCESS: All datasets are valid!")
    else:
        print("FAILURE: Some datasets are invalid!")
    
    all_valid = True
    for name, details in validation_details.items():
        print(f"\n{name} Dataset:")
        print(f"  Loading Status: {'Success' if details['loaded'] else 'Failed'}")
        if details['loaded']:
            print(f"  Memory Usage: {details['memory']:.2f} MB")
            print(f"  Validation Status: {'Valid' if details['valid'] else 'Invalid'}")
        if details['errors']:
            print("  Errors:")
            for error in details['errors']:
                print(f"    - {error}")
            all_valid = False
    
    print("\n" + "="*50)
    if all_valid:
        print("SUCCESS: All datasets are valid!")
    else:
        print("WARNING: Some datasets are invalid. Please review the errors above.")
    print("="*50)

def check_dataset(dataset, dataset_name):
    print(f"\nChecking {dataset_name} Dataset:")
    
    valid = True
    
    # Configuration check
    valid &= check_config_values()
    
    # Required keys check
    valid &= check_required_keys(dataset, dataset_name)
    
    # Only proceed with other checks if required keys exist
    if valid:
        # Dimension checks
        valid &= check_dimensions(dataset, dataset_name)
        
        # Value range checks
        valid &= check_value_ranges(dataset, dataset_name)
        valid &= check_snr_range(dataset, dataset_name, 
                                snr_range=CHANNEL_CONFIG["snr_range"])
        valid &= check_sinr_range(dataset, dataset_name)
        valid &= check_interference_range(dataset, dataset_name)
        
        # Complex value checks
        valid &= check_complex_values(dataset, dataset_name)
        
        # User-related checks
        valid &= check_user_association_dimensions(dataset, dataset_name)
        valid &= check_user_consistency(dataset, dataset_name)
        
        # Invalid value checks
        valid &= check_invalid_values(dataset, dataset_name)
        
        # Print shapes
        print("\nArray Shapes:")
        for key in dataset:
            print(f"  {key}: {dataset[key].shape}")
    
    return valid

def check_config_values():
    """Verify all required configuration values exist"""
    try:
        required_configs = {
            "MIMO_CONFIG": ["tx_antennas", "rx_antennas"],
            "CHANNEL_CONFIG": ["snr_range", "interference"],
            "MULTIUSER_CONFIG": ["num_users"],
            "CONFIG": ["sinr_target"]
        }
        
        for config_name, required_keys in required_configs.items():
            for key in required_keys:
                if key not in eval(config_name):
                    print(f"Error: Missing {key} in {config_name}")
                    return False
        return True
    except NameError as e:
        print(f"Error: Missing configuration: {str(e)}")
        return False

def check_user_consistency(dataset, dataset_name):
    """Verify consistency of user-related data"""
    num_users = MULTIUSER_CONFIG["num_users"]
    
    # Check user association dimensions
    if dataset["user_association"].shape != (dataset["channel_realizations"].shape[0], num_users):
        print(f"Error: Invalid user association dimensions in {dataset_name}")
        return False
    
    # Check if user associations are valid (binary values)
    if not np.all(np.isin(dataset["user_association"], [0, 1])):
        print(f"Error: Invalid user association values in {dataset_name}")
        return False
    
    return True

def check_invalid_values(dataset, dataset_name):
    """Check for NaN or Infinity values"""
    for key, values in dataset.items():
        if np.any(np.isnan(values)):
            print(f"Error: NaN values found in {dataset_name} - {key}")
            return False
        if np.any(np.isinf(values)):
            print(f"Error: Infinite values found in {dataset_name} - {key}")
            return False
    return True
    
def get_dataset_memory_usage(dataset):
    """Calculate memory usage of dataset in MB"""
    memory_usage = sum(arr.nbytes for arr in dataset.values())
    return memory_usage / (1024 * 1024)  # Convert to MB

def main():
    print("Starting dataset validation...")
    
    # Create a dictionary to store detailed validation results
    validation_details = {
        "Training": {"loaded": False, "valid": False, "memory": 0, "errors": []},
        "Validation": {"loaded": False, "valid": False, "memory": 0, "errors": []},
        "Test": {"loaded": False, "valid": False, "memory": 0, "errors": []}
    }
    
    # First check configuration
    if not check_config_values():
        print("ERROR: Invalid configuration values")
        return False
    
    # Load and validate each dataset
    datasets = {
        "Training": TRAINING_DATA_PATH,
        "Validation": VALIDATION_DATA_PATH,
        "Test": TEST_DATA_PATH
    }
    
    all_valid = True
    for name, file_path in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {name} Dataset")
        print(f"{'='*50}")
        
        try:
            # Load dataset
            dataset = load_dataset(file_path)
            if dataset is not None:
                validation_details[name]["loaded"] = True
                validation_details[name]["memory"] = get_dataset_memory_usage(dataset)
                
                # Run all validation checks
                validation_details[name]["valid"] = check_dataset(dataset, name)
                if not validation_details[name]["valid"]:
                    all_valid = False
            else:
                validation_details[name]["errors"].append("Failed to load dataset")
                all_valid = False
                
        except Exception as e:
            validation_details[name]["errors"].append(f"Validation error: {str(e)}")
            all_valid = False
    
    # Print comprehensive summary
    print_validation_summary(validation_details)
    
    return all_valid

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Critical error during validation: {str(e)}")
        exit(1)