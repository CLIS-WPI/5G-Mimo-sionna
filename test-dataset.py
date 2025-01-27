import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind
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

    def test_dataset_distributions(self):
        """Test if training, validation, and test datasets have similar distributions"""
        # Compare SNR distributions
        train_snr = self.datasets["training"]["snr"]
        val_snr = self.datasets["validation"]["snr"]
        test_snr = self.datasets["test"]["snr"]

        # Kolmogorov-Smirnov test for SNR
        ks_stat_train_val, p_value_train_val = ks_2samp(train_snr, val_snr)
        ks_stat_train_test, p_value_train_test = ks_2samp(train_snr, test_snr)
        
        print(f"\nKolmogorov-Smirnov Test Results for SNR:")
        print(f"Training vs. Validation: Statistic={ks_stat_train_val:.4f}, p-value={p_value_train_val:.4f}")
        print(f"Training vs. Test: Statistic={ks_stat_train_test:.4f}, p-value={p_value_train_test:.4f}")

        # t-Test for SNR
        t_stat_train_val, p_value_train_val = ttest_ind(train_snr, val_snr)
        t_stat_train_test, p_value_train_test = ttest_ind(train_snr, test_snr)
        
        print(f"\nt-Test Results for SNR:")
        print(f"Training vs. Validation: Statistic={t_stat_train_val:.4f}, p-value={p_value_train_val:.4f}")
        print(f"Training vs. Test: Statistic={t_stat_train_test:.4f}, p-value={p_value_train_test:.4f}")

        # Compare SINR distributions
        train_sinr = self.datasets["training"]["sinr"].flatten()
        val_sinr = self.datasets["validation"]["sinr"].flatten()
        test_sinr = self.datasets["test"]["sinr"].flatten()

        # Kolmogorov-Smirnov test for SINR
        ks_stat_train_val, p_value_train_val = ks_2samp(train_sinr, val_sinr)
        ks_stat_train_test, p_value_train_test = ks_2samp(train_sinr, test_sinr)
        
        print(f"\nKolmogorov-Smirnov Test Results for SINR:")
        print(f"Training vs. Validation: Statistic={ks_stat_train_val:.4f}, p-value={p_value_train_val:.4f}")
        print(f"Training vs. Test: Statistic={ks_stat_train_test:.4f}, p-value={p_value_train_test:.4f}")

        # t-Test for SINR
        t_stat_train_val, p_value_train_val = ttest_ind(train_sinr, val_sinr)
        t_stat_train_test, p_value_train_test = ttest_ind(train_sinr, test_sinr)
        
        print(f"\nt-Test Results for SINR:")
        print(f"Training vs. Validation: Statistic={t_stat_train_val:.4f}, p-value={p_value_train_val:.4f}")
        print(f"Training vs. Test: Statistic={t_stat_train_test:.4f}, p-value={p_value_train_test:.4f}")

        # Compare Interference distributions
        train_interference = self.datasets["training"]["interference"].flatten()
        val_interference = self.datasets["validation"]["interference"].flatten()
        test_interference = self.datasets["test"]["interference"].flatten()

        # Kolmogorov-Smirnov test for Interference
        ks_stat_train_val, p_value_train_val = ks_2samp(train_interference, val_interference)
        ks_stat_train_test, p_value_train_test = ks_2samp(train_interference, test_interference)
        
        print(f"\nKolmogorov-Smirnov Test Results for Interference:")
        print(f"Training vs. Validation: Statistic={ks_stat_train_val:.4f}, p-value={p_value_train_val:.4f}")
        print(f"Training vs. Test: Statistic={ks_stat_train_test:.4f}, p-value={p_value_train_test:.4f}")

        # t-Test for Interference
        t_stat_train_val, p_value_train_val = ttest_ind(train_interference, val_interference)
        t_stat_train_test, p_value_train_test = ttest_ind(train_interference, test_interference)
        
        print(f"\nt-Test Results for Interference:")
        print(f"Training vs. Validation: Statistic={t_stat_train_val:.4f}, p-value={p_value_train_val:.4f}")
        print(f"Training vs. Test: Statistic={t_stat_train_test:.4f}, p-value={p_value_train_test:.4f}")

        # Assert that p-values are above a significance threshold (e.g., 0.05)
        significance_threshold = 0.05
        self.assertGreater(p_value_train_val, significance_threshold, "Training and validation datasets have significantly different distributions")
        self.assertGreater(p_value_train_test, significance_threshold, "Training and test datasets have significantly different distributions")

        # Generate and save plots
        self._generate_plots(train_snr, val_snr, test_snr, train_sinr, val_sinr, test_sinr, train_interference, val_interference, test_interference)

    def _generate_plots(self, train_snr, val_snr, test_snr, train_sinr, val_sinr, test_sinr, train_interference, val_interference, test_interference):
        """Generate and save distribution and box plots for SNR, SINR, and interference"""
        # Create output directory for plots
        import os
        os.makedirs("./plots", exist_ok=True)

        # Plot SNR distributions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_snr, label="Training SNR", fill=True)
        sns.kdeplot(val_snr, label="Validation SNR", fill=True)
        sns.kdeplot(test_snr, label="Test SNR", fill=True)
        plt.title("SNR Distributions")
        plt.xlabel("SNR")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("./plots/snr_distributions.png")
        plt.close()

        # Plot SINR distributions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_sinr, label="Training SINR", fill=True)
        sns.kdeplot(val_sinr, label="Validation SINR", fill=True)
        sns.kdeplot(test_sinr, label="Test SINR", fill=True)
        plt.title("SINR Distributions")
        plt.xlabel("SINR")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("./plots/sinr_distributions.png")
        plt.close()

        # Plot Interference distributions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_interference, label="Training Interference", fill=True)
        sns.kdeplot(val_interference, label="Validation Interference", fill=True)
        sns.kdeplot(test_interference, label="Test Interference", fill=True)
        plt.title("Interference Distributions")
        plt.xlabel("Interference")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("./plots/interference_distributions.png")
        plt.close()

        # Plot SNR boxplots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[train_snr, val_snr, test_snr], notch=True)
        plt.xticks([0, 1, 2], ["Training", "Validation", "Test"])
        plt.title("SNR Boxplot")
        plt.savefig("./plots/snr_boxplot.png")
        plt.close()

        # Plot SINR boxplots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[train_sinr, val_sinr, test_sinr], notch=True)
        plt.xticks([0, 1, 2], ["Training", "Validation", "Test"])
        plt.title("SINR Boxplot")
        plt.savefig("./plots/sinr_boxplot.png")
        plt.close()

        # Plot Interference boxplots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[train_interference, val_interference, test_interference], notch=True)
        plt.xticks([0, 1, 2], ["Training", "Validation", "Test"])
        plt.title("Interference Boxplot")
        plt.savefig("./plots/interference_boxplot.png")
        plt.close()

if __name__ == '__main__':
    unittest.main(verbosity=2)