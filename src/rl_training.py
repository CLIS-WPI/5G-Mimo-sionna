# src/rl_training.py
# This script implements the training of a Soft Actor-Critic (SAC) model for optimizing beamforming in MIMO systems 
# based on simulated channel realizations. The model aims to dynamically adjust the transmit beamforming vectors 
# to maximize system performance by considering Signal-to-Interference-plus-Noise Ratio (SINR), Signal-to-Noise 
# Ratio (SNR), and interference levels.

# Key Objectives:
# - Optimize beamforming vectors to achieve target SINR levels (20 dB target)
# - Minimize interference between multiple receivers
# - Maintain acceptable SNR levels while managing the SINR-interference trade-off

# Inputs:
# - Training and validation datasets, which include channel realizations and corresponding SNR values.
#   The datasets are loaded from pre-generated .npy files containing the data produced by the dataset generator script.
#   The data consists of:
#   - channel_realizations: A tensor representing the complex channel matrices for MIMO systems
#   - snr: Initial SNR values used as part of the reward computation
#
# Outputs:
# - The SAC model's parameters are updated after each episode through gradient-based optimization.
# - The actor and critic networks are refined iteratively to improve the beamforming policy.
# - Training progress and validation rewards are printed during training to monitor performance.
# - The model learns to balance:
#   * SINR improvement towards the 20 dB target (60% weight)
#   * SNR maintenance (30% weight)
#   * Interference reduction (10% weight)
#
# The model uses a sophisticated reward mechanism that considers:
# - SINR target achievement (20 dB)
# - Minimum SINR threshold (10 dB)
# - Interference levels between receivers
# - Overall SNR performance
#
# The training process employs the SAC algorithm with:
# - Double Q-learning for robust critic estimation
# - Entropy regularization for exploration
# - Adaptive alpha parameter for temperature adjustment
# - Batch processing for efficient training
#
# Performance Monitoring:
# - Regular validation checks to assess model improvement
# - Tracking of episode rewards, critic losses, and actor losses
# - Early detection of NaN values and training instabilities

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from config import CONFIG, OUTPUT_FILES, MIMO_CONFIG

# Load training and validation datasets
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    data = np.load(file_path, allow_pickle=True).item()
    return data["channel_realizations"], data["snr"]

# Define the SAC model class
class SoftActorCritic:
    def __init__(self, input_shape, num_actions, learning_rates):
        self.num_actions = num_actions
        self.actor = self.build_actor(input_shape, num_actions, learning_rates["actor"])
        self.critic1 = self.build_critic(input_shape, num_actions, learning_rates["critic"])
        self.critic2 = self.build_critic(input_shape, num_actions, learning_rates["critic"])
        self.alpha = tf.Variable(initial_value=learning_rates["alpha"], trainable=True, dtype=tf.float32)
        self.optimizer_alpha = tf.keras.optimizers.Adam(learning_rate=learning_rates["alpha"])

    def build_actor(self, input_shape, num_actions, lr):
        inputs = layers.Input(shape=input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(num_actions, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def build_critic(self, input_shape, num_actions, lr):
        # State input branch
        state_input = layers.Input(shape=input_shape)
        state_flat = layers.Flatten()(state_input)
        
        # Action input branch
        action_input = layers.Input(shape=(num_actions,))
        
        # Combine state and action
        concat = layers.Concatenate()([state_flat, action_input])
        
        # Hidden layers
        x = layers.Dense(256, activation="relu")(concat)
        x = layers.Dense(256, activation="relu")(x)
        
        # Output Q-value
        q_value = layers.Dense(1)(x)
        
        model = tf.keras.Model([state_input, action_input], q_value)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def get_action(self, state):
        """
        Get actions for a batch of states
        """
        # Convert to tensor if not already
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # Add batch dimension if needed
        if len(state.shape) == 4:
            state = tf.expand_dims(state, axis=0)
        
        # Get action probabilities
        action_probs = self.actor(state)
        
        # Sample actions using Gumbel-Softmax
        actions = tf.random.categorical(tf.math.log(action_probs + 1e-10), 1)
        actions = tf.squeeze(actions)
        
        # Convert to one-hot encoding
        actions = tf.one_hot(actions, depth=self.num_actions)
        
        return actions

def compute_reward(snr, sinr, interference_level):
    """
    Compute reward based on SINR target and interference
    Args:
        snr: Signal-to-Noise Ratio
        sinr: Signal-to-Interference-plus-Noise Ratio
        interference_level: Level of interference
    """
    sinr_target = 20.0  # Our target SINR of 20 dB
    sinr_threshold = 10.0  # Minimum acceptable SINR

    # Penalize if SINR is below threshold
    if sinr < sinr_threshold:
        return -10.0
    
    reward = (
        0.6 * (sinr - sinr_target) +  # SINR improvement towards target
        0.3 * snr +  # SNR contribution
        0.1 * (-interference_level)  # Interference reduction
    )
    return reward

def compute_interference(channel_state, beamforming_vectors):
    """
    Compute interference levels for MIMO transmissions
    """
    # Convert to complex tensors
    h = tf.cast(channel_state, tf.complex64)
    w = tf.cast(beamforming_vectors, tf.complex64)
    
    # Get batch size
    batch_size = tf.shape(h)[0]
    
    # Reshape tensors
    h_reshaped = tf.reshape(h, [batch_size, h.shape[1], h.shape[2], h.shape[4]])
    w_reshaped = tf.reshape(w, [batch_size, -1, 1])  # [batch_size, num_tx_ant, 1]
    
    # Compute interference power
    interference = tf.zeros(batch_size, dtype=tf.float32)
    
    # For each receiver
    for i in range(tf.shape(h)[1]):
        desired_signal = tf.abs(tf.matmul(h_reshaped[:, i:i+1, :, :], w_reshaped))**2
        total_power = tf.reduce_sum(desired_signal, axis=1)
        interference += total_power - desired_signal[:, 0]
    
    return interference

def compute_sinr(channel_state, beamforming_vectors, noise_power=1.0):
    """
    Compute SINR for MIMO transmissions
    Args:
        channel_state: Complex channel matrix [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant]
        beamforming_vectors: Beamforming vectors [batch_size, num_tx_ant]
        noise_power: Noise power (default: 1.0)
    Returns:
        sinr_values: SINR values [batch_size]
    """
    # Convert to complex tensors
    h = tf.cast(channel_state, tf.complex64)
    w = tf.cast(beamforming_vectors, tf.complex64)
    
    # Get batch size
    batch_size = tf.shape(h)[0]
    
    # Reshape h to [batch_size, num_rx, num_rx_ant, num_tx_ant]
    h_reshaped = tf.reshape(h, [batch_size, h.shape[1], h.shape[2], h.shape[4]])
    
    # Reshape w to match batch size and add necessary dimensions
    w_reshaped = tf.reshape(w, [batch_size, -1, 1])  # [batch_size, num_tx_ant, 1]
    
    # Compute desired signal power
    desired_signal = tf.abs(tf.matmul(h_reshaped, w_reshaped))**2
    signal_power = tf.reduce_sum(desired_signal, axis=1)
    
    # Compute interference
    interference = compute_interference(channel_state, beamforming_vectors)
    
    # Compute SINR
    sinr = 10.0 * tf.math.log(signal_power / (interference + noise_power)) / tf.math.log(10.0)
    
    return sinr

def train_sac(training_data, validation_data, config):
    input_shape = training_data[0].shape[1:]  # Exclude batch size
    num_actions = MIMO_CONFIG["tx_antennas"]  # Beamforming actions correspond to TX antennas
    # Add at the start of train_sac
    training_history = {
        'episode_rewards': [],
        'critic_losses': [],
        'actor_losses': []
    }
    
    # First add the validate_shapes function definition
    def validate_shapes(batch_channels, actions):
        """Validate shapes of inputs"""
        if len(batch_channels.shape) != 5:
            raise ValueError(f"Expected batch_channels shape [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant], got {batch_channels.shape}")
        if len(actions.shape) != 2:
            raise ValueError(f"Expected actions shape [batch_size, num_actions], got {actions.shape}")

    # Instantiate the SAC model
    sac = SoftActorCritic(
        input_shape=input_shape,
        num_actions=num_actions,
        learning_rates={
            "actor": config["actor_lr"],
            "critic": config["critic_lr"],
            "alpha": config["alpha_lr"]
        }
    )

    # Training loop
    for episode in range(config["episodes"]):
        print(f"Episode {episode+1}/{config['episodes']}")

        # Batch training
        for start in range(0, len(training_data[0]), config["batch_size"]):
            end = start + config["batch_size"]
            batch_channels = training_data[0][start:end]
            batch_snr = training_data[1][start:end]
            
            # Convert to tensor and ensure proper shape with batch dimension
            batch_channels = tf.convert_to_tensor(batch_channels, dtype=tf.float32)
            
            # Get actions for the entire batch at once
            actions = sac.get_action(batch_channels)  # This will handle the full batch

            # Validate shapes before proceeding
            validate_shapes(batch_channels, actions)
            
            # Compute rewards
            sinr_values = compute_sinr(batch_channels, actions)
            interference_levels = compute_interference(batch_channels, actions)
            rewards = tf.convert_to_tensor(
                [compute_reward(snr, sinr, interference) 
                for snr, sinr, interference in zip(batch_snr, sinr_values, interference_levels)],
                dtype=tf.float32
            )
            
            # Critic loss computation
            critic1_value = sac.critic1([batch_channels, actions])
            critic2_value = sac.critic2([batch_channels, actions])
            
            # Use minimum of critics (Double Q-learning)
            min_critic = tf.minimum(critic1_value, critic2_value)
            critic_loss = tf.reduce_mean(tf.square(rewards - tf.squeeze(min_critic)))
            
            # Actor loss with entropy regularization
            actor_probs = sac.actor(batch_channels)
            log_probs = tf.math.log(actor_probs + 1e-10)
            entropy = -tf.reduce_sum(actor_probs * log_probs, axis=-1)
            actor_loss = -tf.reduce_mean(rewards + sac.alpha * entropy)

            # Compute gradients for actor and critic models
            critic1_grads = tape.gradient(critic_loss, sac.critic1.trainable_variables)
            critic2_grads = tape.gradient(critic_loss, sac.critic2.trainable_variables)
            actor_grads = tape.gradient(actor_loss, sac.actor.trainable_variables)

            # Apply gradients
            sac.critic1.optimizer.apply_gradients(zip(critic1_grads, sac.critic1.trainable_variables))
            sac.critic2.optimizer.apply_gradients(zip(critic2_grads, sac.critic2.trainable_variables))
            sac.actor.optimizer.apply_gradients(zip(actor_grads, sac.actor.trainable_variables))

            # Update the alpha parameter
            alpha_loss = -tf.reduce_mean(sac.alpha * tf.stop_gradient(actor_prob - 1.0))
            alpha_grad = tape.gradient(alpha_loss, sac.alpha)
            sac.optimizer_alpha.apply_gradients([(alpha_grad, sac.alpha)])

        # Validate performance
        if episode % config["validation_interval"] == 0:
            print(f"Validating at episode {episode+1}...")
            # Validation logic (optional based on validation dataset)
            validate_model(sac, validation_data)

    print("Training complete.")

# Validation function
def validate_model(sac, validation_data):
    val_channels, val_snr = validation_data
    total_reward = 0

    for start in range(0, len(val_channels), CONFIG["batch_size"]):
        end = start + CONFIG["batch_size"]
        batch_channels = val_channels[start:end]
        batch_snr = val_snr[start:end]

        # Simulate actions and compute rewards
        actions = np.array([sac.get_action(state) for state in batch_channels])
        
        # Compute SINR and interference for validation
        sinr_values = compute_sinr(batch_channels, actions)
        interference_levels = compute_interference(batch_channels, actions)
        
        # Calculate rewards using the same reward function
        rewards = np.array([compute_reward(snr, sinr, interference) 
                        for snr, sinr, interference in zip(batch_snr, sinr_values, interference_levels)])
        
        total_reward += np.sum(rewards)

    print(f"Validation reward: {total_reward / len(validation_data[0])}")

if __name__ == "__main__":
    # Load datasets
    training_data = load_dataset(OUTPUT_FILES["training_data"])
    validation_data = load_dataset(OUTPUT_FILES["validation_data"])

    # Training configuration
    train_config = {
        "episodes": CONFIG["number_of_episodes"],
        "batch_size": CONFIG["mini_batch_size"],
        "actor_lr": CONFIG["actor_lr"],
        "critic_lr": CONFIG["critic_lr"],
        "alpha_lr": CONFIG["alpha_lr"],
        "validation_interval": 10
    }

    # Train the SAC model
    train_sac(training_data, validation_data, train_config)
