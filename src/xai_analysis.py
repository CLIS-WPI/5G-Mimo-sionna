# src/xai_analysis.py
"""
src/xai_analysis.py

Purpose:
This script implements Explainable AI (XAI) analysis for the trained SAC beamforming model using SHAP 
(SHapley Additive exPlanations) values. It helps understand how different features influence the model's 
beamforming decisions in MIMO systems.

Key Objectives:
- Analyze how channel conditions affect beamforming decisions
- Understand the impact of SNR and interference on beam selection
- Visualize feature importance in the decision-making process
- Provide interpretable insights into the model's behavior

Inputs:
1. Trained SAC Model:
- Actor network weights
- Critic network weights
- Model architecture parameters

2. Test Dataset:
- Channel matrices (complex-valued MIMO channel realizations)
- SNR values
- Interference measurements
- Performance metrics (SINR, throughput)

Outputs:
1. Feature Importance Analysis:
- Visualization of how different features (channel magnitude, phase, SNR, interference)
influence beamforming decisions
- Relative importance scores for each feature

2. SHAP Summary Plots:
- Global model interpretation showing feature impacts
- Individual decision analysis for specific cases
- Feature interaction visualizations

3. Decision Factor Analysis:
- Detailed breakdown of factors influencing specific beamforming decisions
- Trade-off analysis between different optimization objectives
- Channel contribution vs. interference management decisions

Key Components:
1. BeamformingXAIAnalyzer:
- Loads and processes the trained model
- Generates SHAP values
- Creates visualizations and analysis reports

2. Analysis Methods:
- Feature importance calculation
- Decision factor analysis
- SHAP value generation and interpretation
- Visualization utilities

Usage:
1. Train the SAC model using rl_training.py
2. Run this script to analyze the trained model
3. Check results in the results/xai_analysis directory

The analysis helps understand:
- Which features most strongly influence beamforming decisions
- How the model balances different objectives (SINR, interference, SNR)
- When and why specific beamforming patterns are chosen
- The model's adaptation to different channel conditions
"""


import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import os

from config import CONFIG, MIMO_CONFIG
from rl_training import load_dataset, SoftActorCritic

class BeamformingXAIAnalyzer:
    def __init__(self, model_path: str):
        """Initialize the XAI analyzer with a trained model."""
        self.input_shape = (MIMO_CONFIG['tx_antennas'], MIMO_CONFIG['rx_antennas'])
        self.num_actions = MIMO_CONFIG['tx_antennas']
        
        # Initialize SAC model
        learning_rates = {
            'actor': CONFIG['actor_lr'],
            'critic': CONFIG['critic_lr'],
            'alpha': CONFIG['alpha_lr']
        }
        self.model = SoftActorCritic(self.input_shape, self.num_actions, learning_rates)
        
        # Load trained weights
        self.model.load_weights(model_path)
        
    def prepare_background_data(self, data: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """Prepare background data for SHAP analysis."""
        indices = np.random.choice(len(data), num_samples, replace=False)
        return data[indices]
    
    def generate_shap_values(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate SHAP values for the given data."""
        # Create explainer using the actor network
        background = self.prepare_background_data(data)
        explainer = shap.DeepExplainer(self.model.actor, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data)
        return shap_values, background
    
    def analyze_feature_importance(self, shap_values: np.ndarray) -> dict:
        """Analyze feature importance based on SHAP values."""
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create dictionary of feature importance
        importance_dict = {
            'channel_magnitude': feature_importance[0],
            'channel_phase': feature_importance[1],
            'snr_impact': feature_importance[2],
            'interference_impact': feature_importance[3]
        }
        
        return importance_dict
    
    def plot_feature_importance(self, importance_dict: dict, save_path: str):
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        sns.barplot(x=values, y=features)
        plt.title('Feature Importance in Beamforming Decisions')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_shap_summary(self, shap_values: np.ndarray, features: np.ndarray, save_path: str):
        """Create SHAP summary plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, features, show=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def analyze_decision_factors(self, test_sample: np.ndarray) -> dict:
        """Analyze factors influencing a specific beamforming decision."""
        # Get model prediction
        action = self.model.get_action(test_sample)
        
        # Calculate SHAP values for this sample
        explainer = shap.DeepExplainer(self.model.actor, test_sample[np.newaxis, ...])
        shap_values = explainer.shap_values(test_sample[np.newaxis, ...])[0]
        
        # Analyze decision factors
        factors = {
            'predicted_action': action,
            'channel_contribution': np.mean(shap_values[:, :2]),
            'snr_contribution': np.mean(shap_values[:, 2]),
            'interference_contribution': np.mean(shap_values[:, 3])
        }
        
        return factors

def main():
    # Load test data
    test_data, test_snr = load_dataset(CONFIG['test_data_path'])
    
    # Initialize XAI analyzer
    analyzer = BeamformingXAIAnalyzer('path_to_trained_model')
    
    # Generate SHAP values
    shap_values, background = analyzer.generate_shap_values(test_data)
    
    # Analyze feature importance
    importance_dict = analyzer.analyze_feature_importance(shap_values)
    
    # Create output directory
    os.makedirs('results/xai_analysis', exist_ok=True)
    
    # Plot and save results
    analyzer.plot_feature_importance(
        importance_dict, 
        'results/xai_analysis/feature_importance.png'
    )
    
    analyzer.plot_shap_summary(
        shap_values, 
        test_data, 
        'results/xai_analysis/shap_summary.png'
    )
    
    # Analyze specific test case
    test_case = test_data[0]
    decision_factors = analyzer.analyze_decision_factors(test_case)
    print("Decision Factors Analysis:")
    for factor, value in decision_factors.items():
        print(f"{factor}: {value}")

if __name__ == "__main__":
    main()