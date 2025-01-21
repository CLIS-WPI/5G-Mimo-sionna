import unittest
import numpy as np
from src.config import CONFIG, MIMO_CONFIG, CHANNEL_CONFIG, MULTIUSER_CONFIG, RESOURCE_GRID

class TestDatasetGeneration(unittest.TestCase):
    """Test suite for dataset generation validation"""
    
    @classmethod
    def setUpClass(cls):
        """Load all datasets once before running tests"""
        cls.datasets = {
            "training": np.load("./data/training/training_data.npy", allow_pickle=True).item(),
            "validation": np.load("./data/validation/validation_data.npy", allow_pickle=True).item(),
            "test": np.load("./data/test/test_data.npy", allow_pickle=True).item()
        }

    def test_required_keys(self):
        """Test presence of all required keys in datasets"""
        required_keys = [
            "channel_realizations", 
            "snr", 
            "sinr", 
            "interference",
            "user_association",
            "precoding_matrices"
        ]
        
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                for key in required_keys:
                    self.assertIn(key, dataset, f"Missing key '{key}' in {name} dataset")

    def test_dimensions(self):
        """Test dimensions of dataset arrays"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                num_samples = dataset["channel_realizations"].shape[0]
                num_users = MULTIUSER_CONFIG["num_users"]
                
                # Print detailed information for debugging
                print(f"\nDetailed dimension analysis for {name} dataset:")
                print(f"Actual shape: {dataset['channel_realizations'].shape}")
                
                # Expected shapes for each array
                channel_shape = (
                    num_samples,
                    num_users,
                    MIMO_CONFIG["rx_antennas"],
                    RESOURCE_GRID["ofdm_symbols"],
                    RESOURCE_GRID["subcarriers"]
                )
                
                snr_shape = (num_samples,)
                sinr_shape = (num_samples, num_users)
                interference_shape = (num_samples, num_users)
                precoding_shape = (num_samples, num_users, MIMO_CONFIG["rx_antennas"])
                
                # Test each array's shape
                self.assertEqual(
                    dataset["channel_realizations"].shape,
                    channel_shape,
                    f"Invalid channel_realizations shape in {name}"
                )
                
                self.assertEqual(
                    dataset["snr"].shape,
                    snr_shape,
                    f"Invalid SNR shape in {name}"
                )
                
                self.assertEqual(
                    dataset["sinr"].shape,
                    sinr_shape,
                    f"Invalid SINR shape in {name}"
                )
                
                self.assertEqual(
                    dataset["interference"].shape,
                    interference_shape,
                    f"Invalid interference shape in {name}"
                )
                
                self.assertEqual(
                    dataset["precoding_matrices"].shape,
                    precoding_shape,
                    f"Invalid precoding matrices shape in {name}"
                )

    def test_value_ranges(self):
        """Test value ranges for different metrics"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                # SNR range
                self.assertTrue(
                    np.all(dataset["snr"] >= CHANNEL_CONFIG["snr_range"][0]) and 
                    np.all(dataset["snr"] <= CHANNEL_CONFIG["snr_range"][1]),
                    f"SNR values out of range in {name}"
                )
                
                # Interference range
                self.assertTrue(
                    np.all(dataset["interference"] >= CHANNEL_CONFIG["interference"]["interference_power_range"][0]) and 
                    np.all(dataset["interference"] <= CHANNEL_CONFIG["interference"]["interference_power_range"][1]),
                    f"Interference values out of range in {name}"
                )

    def test_complex_values(self):
        """Test complex-valued data properties"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                self.assertTrue(
                    np.iscomplexobj(dataset["channel_realizations"]),
                    f"Channel realizations not complex in {name}"
                )
                self.assertTrue(
                    np.iscomplexobj(dataset["precoding_matrices"]),
                    f"Precoding matrices not complex in {name}"
                )

    def test_data_integrity(self):
        """Test for NaN and Inf values"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                for key, values in dataset.items():
                    if isinstance(values, np.ndarray):
                        self.assertFalse(
                            np.any(np.isnan(values)),
                            f"NaN values found in {name} - {key}"
                        )
                        self.assertFalse(
                            np.any(np.isinf(values)),
                            f"Infinite values found in {name} - {key}"
                        )

    def test_user_association(self):
        """Test user association properties"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                user_assoc = dataset["user_association"]
                self.assertTrue(
                    np.all(np.isin(user_assoc, [0, 1])),
                    f"Invalid user association values in {name}"
                )

if __name__ == '__main__':
    unittest.main(verbosity=2)