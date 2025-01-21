import unittest
import numpy as np
from src.config import CONFIG, MIMO_CONFIG, CHANNEL_CONFIG, MULTIUSER_CONFIG

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
                
                # Print actual and expected shapes for debugging
                actual_shape = dataset["channel_realizations"].shape
                expected_shape = (num_samples, num_users, MIMO_CONFIG["rx_antennas"], 
                                1, MIMO_CONFIG["tx_antennas"], 1, 1)
                
                print(f"\nDetailed dimension analysis for {name} dataset:")
                print(f"Actual shape  : {actual_shape}")
                print(f"Expected shape: {expected_shape}")
                print("\nShape components:")
                print(f"- Number of samples: {num_samples}")
                print(f"- Number of users: {num_users}")
                print(f"- RX antennas: {MIMO_CONFIG['rx_antennas']}")
                print(f"- TX antennas: {MIMO_CONFIG['tx_antennas']}")
                
                # Check each dimension separately
                try:
                    self.assertEqual(actual_shape[0], expected_shape[0], 
                        f"Number of samples mismatch in {name}")
                    self.assertEqual(actual_shape[1], expected_shape[1], 
                        f"Number of users mismatch in {name}")
                    self.assertEqual(actual_shape[2], expected_shape[2], 
                        f"RX antennas mismatch in {name}")
                    
                    # Print additional information about the problematic dimensions
                    print("\nProblematic dimensions:")
                    print(f"Dimension 4: Expected {expected_shape[3]}, Got {actual_shape[3]}")
                    print(f"Dimension 5: Expected {expected_shape[4]}, Got {actual_shape[4]}")
                    if len(actual_shape) != len(expected_shape):
                        print(f"Dimension count mismatch: Expected {len(expected_shape)}, Got {len(actual_shape)}")
                
                except AssertionError as e:
                    print(f"\nAssertion Error in {name} dataset:")
                    print(str(e))
                
                # Check other array dimensions
                print("\nOther array shapes:")
                print(f"SNR shape: {dataset['snr'].shape}")
                print(f"SINR shape: {dataset['sinr'].shape}")
                print(f"Interference shape: {dataset['interference'].shape}")
                print(f"Precoding matrices shape: {dataset['precoding_matrices'].shape}")
                
                # Final shape assertion
                self.assertEqual(
                    actual_shape,
                    expected_shape,
                    f"Invalid channel_realizations shape in {name}"
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