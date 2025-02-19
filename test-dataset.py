import os
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
        try:
            for split in ['training', 'validation', 'test']:
                path = f"./data/{split}/{split}_data.npy"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Dataset file not found: {path}")
                
            cls.datasets = {
                "training": np.load("./data/training/training_data.npy", allow_pickle=True).item(),
                "validation": np.load("./data/validation/validation_data.npy", allow_pickle=True).item(),
                "test": np.load("./data/test/test_data.npy", allow_pickle=True).item()
            }
            
            # Verify data is not empty
            for name, dataset in cls.datasets.items():
                if not dataset or len(dataset) == 0:
                    raise ValueError(f"Dataset {name} is empty")
                
            print("Successfully loaded datasets:")
            for name, dataset in cls.datasets.items():
                print(f"{name}: {dataset['channel_realizations'].shape}")
                
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise

    def test_required_keys(self):
        """Test presence of all required keys in datasets"""
        required_keys = [
            "channel_realizations", 
            "snr", 
            "sinr", 
            "interference",
            "user_association",
            "precoding_matrices",
            "doppler_info"
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
                num_users = CONFIG["num_users"]  # Changed from MULTIUSER_CONFIG to CONFIG
                
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
                # Update precoding shape to match new structure
                precoding_shape = (num_samples, MIMO_CONFIG["tx_antennas"], MIMO_CONFIG["rx_antennas"])
                user_association_shape = (num_samples, num_users)
                
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
    def test_doppler_info(self):
        """Test doppler information structure and values"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                # Check doppler_info structure
                self.assertIn("doppler_info", dataset)
                doppler_info = dataset["doppler_info"]
                
                # Check required fields
                required_fields = ["max_speed", "min_speed", "carrier_frequency"]
                for field in required_fields:
                    self.assertIn(field, doppler_info)
                
                # Validate values
                self.assertGreaterEqual(doppler_info["max_speed"], doppler_info["min_speed"])
                self.assertGreaterEqual(doppler_info["min_speed"], 0)
                self.assertGreater(doppler_info["carrier_frequency"], 0)
                
                # Check against config values
                self.assertEqual(doppler_info["max_speed"], 
                            CHANNEL_CONFIG["doppler_shift"]["max_speed"])
                self.assertEqual(doppler_info["min_speed"], 
                            CHANNEL_CONFIG["doppler_shift"]["min_speed"])
                self.assertEqual(doppler_info["carrier_frequency"], 
                            CHANNEL_CONFIG["doppler_shift"]["carrier_frequency"])
            
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
    def test_channel_data_generation(self):
        """Verify channel data generation properties"""
        try:
            print("\nChannel Data Generation Test Results:")
            print("=" * 50)
            
            # Create plot directories first
            try:
                os.makedirs("./plots/channel", exist_ok=True)
            except OSError as e:
                print(f"Error creating plot directories: {str(e)}")
                raise
                
            # Generate plots with error handling
            try:
                self._plot_channel_properties()
                self._plot_correlation_matrices()
            except Exception as e:
                print(f"Error generating plots: {str(e)}")
                raise

            for name, dataset in self.datasets.items():
                with self.subTest(dataset=name):
                    try:
                        channel_data = dataset["channel_realizations"]
                        
                        # Calculate and print statistics
                        real_parts = np.real(channel_data)
                        imag_parts = np.imag(channel_data)
                        
                        print(f"\nDataset: {name}")
                        print(f"Real part - Mean: {np.mean(real_parts):.4f}, Std: {np.std(real_parts):.4f}")
                        print(f"Imaginary part - Mean: {np.mean(imag_parts):.4f}, Std: {np.std(imag_parts):.4f}")
                        
                        # Test for normal distribution using Kolmogorov-Smirnov test
                        try:
                            _, p_value_real = ks_2samp(real_parts.flatten(), 
                                                    np.random.normal(0, 1, size=1000))
                            _, p_value_imag = ks_2samp(imag_parts.flatten(), 
                                                    np.random.normal(0, 1, size=1000))
                        except Exception as e:
                            print(f"Error in statistical tests for {name}: {str(e)}")
                            raise
                        
                        print(f"Normality test p-values - Real: {p_value_real:.4f}, Imaginary: {p_value_imag:.4f}")
                        
                        self.assertGreater(p_value_real, 0.05, 
                                        f"Real parts not normally distributed in {name}")
                        self.assertGreater(p_value_imag, 0.05, 
                                        f"Imaginary parts not normally distributed in {name}")
                        
                        # Test power normalization
                        try:
                            channel_power = np.mean(np.abs(channel_data)**2)
                            print(f"Channel power: {channel_power:.4f}")
                        except Exception as e:
                            print(f"Error calculating channel power for {name}: {str(e)}")
                            raise
                        
                        self.assertAlmostEqual(channel_power, 1.0, delta=0.1,
                                            msg=f"Channel power not normalized in {name}")
                        
                        # Generate and save plots
                        try:
                            plt.figure(figsize=(12, 5))
                            plt.subplot(1, 2, 1)
                            sns.histplot(real_parts.flatten(), stat='density', label='Real')
                            sns.kdeplot(np.random.normal(0, 1, 1000), label='Normal Dist')
                            plt.title(f"{name} - Real Part Distribution")
                            plt.legend()
                            
                            plt.subplot(1, 2, 2)
                            sns.histplot(imag_parts.flatten(), stat='density', label='Imaginary')
                            sns.kdeplot(np.random.normal(0, 1, 1000), label='Normal Dist')
                            plt.title(f"{name} - Imaginary Part Distribution")
                            plt.legend()
                            plt.savefig(f"./plots/channel/{name}_channel_distribution.png")
                            plt.close()
                        except Exception as e:
                            print(f"Error generating plots for {name}: {str(e)}")
                            plt.close()  # Ensure figure is closed even if there's an error
                            raise
                            
                    except KeyError as e:
                        print(f"Missing key in dataset {name}: {str(e)}")
                        raise
                    except Exception as e:
                        print(f"Error processing dataset {name}: {str(e)}")
                        raise
                        
        except Exception as e:
            print(f"Fatal error in test_channel_data_generation: {str(e)}")
            raise
        finally:
            # Ensure all plots are closed
            plt.close('all')

    def test_data_normalization(self):
        """Check for proper data normalization"""
        print("\nData Normalization Test Results:")
        print("=" * 50)
        
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                # Check channel normalization
                channel_data = dataset["channel_realizations"]
                channel_mean = np.mean(np.abs(channel_data))
                channel_std = np.std(np.abs(channel_data))
                
                print(f"\nDataset: {name}")
                print(f"Channel magnitude - Mean: {channel_mean:.4f}, Std: {channel_std:.4f}")
                
                self.assertLess(channel_mean, 1.0, 
                            f"Channel magnitude mean too large in {name}")
                self.assertLess(channel_std, 1.0, 
                            f"Channel magnitude std too large in {name}")
                
                # Check precoding matrix normalization
                precoding = dataset["precoding_matrices"]
                precoding_power = np.mean(np.abs(precoding)**2, axis=-1)
                print(f"Precoding power - Mean: {np.mean(precoding_power):.4f}, "
                    f"Std: {np.std(precoding_power):.4f}")
                
                self.assertTrue(np.allclose(precoding_power, 1.0, atol=0.1),
                            f"Precoding matrices not properly normalized in {name}")
                
                # Generate and save normalization plots
                os.makedirs("./plots/normalization", exist_ok=True)
                
                # Plot channel magnitude distribution
                plt.figure(figsize=(8, 6))
                sns.histplot(np.abs(channel_data).flatten(), stat='density')
                plt.axvline(x=1.0, color='r', linestyle='--', label='Reference')
                plt.title(f"{name} - Channel Magnitude Distribution")
                plt.legend()
                plt.savefig(f"./plots/normalization/{name}_channel_magnitude.png")
                plt.close()
                
                # Plot precoding power distribution
                plt.figure(figsize=(8, 6))
                sns.histplot(precoding_power.flatten(), stat='density')
                plt.axvline(x=1.0, color='r', linestyle='--', label='Expected Power')
                plt.title(f"{name} - Precoding Power Distribution")
                plt.legend()
                plt.savefig(f"./plots/normalization/{name}_precoding_power.png")
                plt.close()

    def test_sinr_calculations(self):
        """Validate SINR calculations"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                sinr = dataset["sinr"]
                snr = dataset["snr"]
                interference = dataset["interference"]
                
                # Test SINR bounds
                self.assertTrue(np.all(sinr <= snr), 
                            f"SINR values exceed SNR in {name}")
                
                # Verify SINR calculation
                # SINR = SNR / (1 + Interference)
                snr_linear = 10**(snr/10)
                interference_linear = 10**(interference/10)
                calculated_sinr = 10 * np.log10(snr_linear / (1 + interference_linear))
                
                # Compare with dataset SINR values
                np.testing.assert_array_almost_equal(sinr, calculated_sinr, decimal=1,
                                                err_msg=f"SINR calculation mismatch in {name}")
                
                # Test SINR requirements
                self.assertTrue(np.mean(sinr) <= CONFIG["sinr_target"],
                            f"Average SINR exceeds target in {name}")
                self.assertTrue(np.all(sinr >= MIMO_CONFIG["sinr_threshold"]),
                            f"SINR values below threshold in {name}")

    def test_statistical_properties(self):
        """Test statistical properties of the channel data"""
        print("\nStatistical Properties Test Results:")
        print("=" * 50)
        
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                channel_data = dataset["channel_realizations"]
                
                # Test spatial correlation
                spatial_correlation = np.corrcoef(
                    np.abs(channel_data.reshape(-1, MIMO_CONFIG["rx_antennas"])).T
                )
                avg_correlation = np.mean(np.abs(spatial_correlation - np.eye(MIMO_CONFIG["rx_antennas"])))
                
                print(f"\nDataset: {name}")
                print(f"Average spatial correlation: {avg_correlation:.4f}")
                
                self.assertLess(avg_correlation, 0.3,
                            f"High spatial correlation in {name}")
                
                # Test temporal correlation
                temporal_correlation = np.corrcoef(
                    np.abs(channel_data.reshape(-1, RESOURCE_GRID["ofdm_symbols"])).T
                )
                avg_temporal_corr = np.mean(np.abs(temporal_correlation - np.eye(RESOURCE_GRID["ofdm_symbols"])))
                print(f"Average temporal correlation: {avg_temporal_corr:.4f}")
                
                self.assertLess(avg_temporal_corr, 0.3,
                            f"High temporal correlation in {name}")
    
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

    def _plot_channel_properties(self):
        """Generate plots for channel data properties"""
        os.makedirs("./plots/channel", exist_ok=True)
        
        for name, dataset in self.datasets.items():
            channel_data = dataset["channel_realizations"]
            
            # Plot real and imaginary distributions
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(np.real(channel_data).flatten(), stat='density', label='Real')
            sns.kdeplot(np.random.normal(0, 1, 1000), label='Normal Dist')
            plt.title(f"{name} - Real Part Distribution")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            sns.histplot(np.imag(channel_data).flatten(), stat='density', label='Imaginary')
            sns.kdeplot(np.random.normal(0, 1, 1000), label='Normal Dist')
            plt.title(f"{name} - Imaginary Part Distribution")
            plt.legend()
            plt.savefig(f"./plots/channel/{name}_channel_distribution.png")
            plt.close()
            
            # Plot channel power
            plt.figure(figsize=(8, 6))
            channel_power = np.mean(np.abs(channel_data)**2, axis=(-2, -1))
            sns.histplot(channel_power.flatten(), stat='density')
            plt.axvline(x=1.0, color='r', linestyle='--', label='Expected Power')
            plt.title(f"{name} - Channel Power Distribution")
            plt.savefig(f"./plots/channel/{name}_power_distribution.png")
            plt.close()

    def _plot_correlation_matrices(self):
        """Generate correlation matrix plots"""
        os.makedirs("./plots/correlation", exist_ok=True)
        
        for name, dataset in self.datasets.items():
            channel_data = dataset["channel_realizations"]
            
            # Spatial correlation
            spatial_corr = np.corrcoef(
                np.abs(channel_data.reshape(-1, MIMO_CONFIG["rx_antennas"])).T
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(spatial_corr, annot=True, cmap='coolwarm')
            plt.title(f"{name} - Spatial Correlation Matrix")
            plt.savefig(f"./plots/correlation/{name}_spatial_correlation.png")
            plt.close()
            
            # Temporal correlation
            temporal_corr = np.corrcoef(
                np.abs(channel_data.reshape(-1, RESOURCE_GRID["ofdm_symbols"])).T
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(temporal_corr, annot=True, cmap='coolwarm')
            plt.title(f"{name} - Temporal Correlation Matrix")
            plt.savefig(f"./plots/correlation/{name}_temporal_correlation.png")
            plt.close()

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
        
    def print_test_summary(self):
        """Print summary of all test results"""
        print("\nTest Summary Report")
        print("=" * 50)
        self._plot_channel_properties()
        self._plot_correlation_matrices()

        for name, dataset in self.datasets.items():
            print(f"\nDataset: {name}")
            print("-" * 30)
            print(f"Number of samples: {len(dataset['channel_realizations'])}")
            print(f"SNR range: [{np.min(dataset['snr']):.2f}, {np.max(dataset['snr']):.2f}] dB")
            print(f"SINR range: [{np.min(dataset['sinr']):.2f}, {np.max(dataset['sinr']):.2f}] dB")
            print(f"Interference range: [{np.min(dataset['interference']):.2f}, "
                f"{np.max(dataset['interference']):.2f}] dB")
            print(f"Channel power mean: {np.mean(np.abs(dataset['channel_realizations'])**2):.4f}")
            
            # Add Doppler information
            doppler_info = dataset['doppler_info']
            print("\nDoppler Information:")
            print(f"Carrier Frequency: {doppler_info['carrier_frequency']/1e9:.2f} GHz")
            print(f"Speed Range: [{doppler_info['min_speed']:.1f}, "
                f"{doppler_info['max_speed']:.1f}] m/s")
    
    def test_channel_normalization(self):
        """Test if channels are properly normalized"""
        for name, dataset in self.datasets.items():
            with self.subTest(dataset=name):
                channels = dataset["channel_realizations"]
                channel_power = np.mean(np.abs(channels)**2)
                
                # Test if power is close to 1
                self.assertAlmostEqual(
                    channel_power, 
                    1.0, 
                    delta=0.1,
                    msg=f"Channel power not properly normalized in {name}"
                )    

if __name__ == '__main__':
    try:
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetGeneration)
        
        # Use a test runner with higher verbosity
        runner = unittest.TextTestRunner(verbosity=3)
        
        print("Starting test execution...")
        result = runner.run(suite)
        
        print(f"\nTests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.wasSuccessful():
            print("\nAll tests passed successfully!")
            
            # Create test instance for visualization
            test_instance = TestDatasetGeneration()
            TestDatasetGeneration.setUpClass()
            
            print("\nGenerating plots...")
            test_instance.print_test_summary()
            print("\nPlots generated successfully")
        else:
            print("\nTest execution failed!")
            for failure in result.failures:
                print(f"\nFailure in {failure[0]}")
                print(failure[1])
            for error in result.errors:
                print(f"\nError in {error[0]}")
                print(error[1])
            
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')