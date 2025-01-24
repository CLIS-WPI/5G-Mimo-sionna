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
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from config import CONFIG, OUTPUT_FILES, MIMO_CONFIG
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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
        
        # Reshape input to combine last two dimensions
        x = layers.Reshape((-1, input_shape[-2] * input_shape[-1]))(inputs)
        
        # Dense layers
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        
        # Output layer
        outputs = layers.Dense(num_actions, activation="softmax")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def build_critic(self, input_shape, num_actions, lr):
        # State input branch
        state_input = layers.Input(shape=input_shape)
        x = layers.Reshape((-1, input_shape[-2] * input_shape[-1]))(state_input)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Flatten()(x)
        
        # Action input branch
        action_input = layers.Input(shape=(num_actions,))
        
        # Combine state and action
        concat = layers.Concatenate()([x, action_input])
        
        # Dense layers
        x = layers.Dense(256, activation="relu")(concat)
        x = layers.Dense(128, activation="relu")(x)
        
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
    Compute reward with better scaling and positive baseline
    """
    sinr_target = tf.constant(20.0, dtype=tf.float32)
    sinr_threshold = tf.constant(10.0, dtype=tf.float32)

    # Convert inputs to tensors
    snr = tf.cast(snr, tf.float32)
    sinr = tf.cast(sinr, tf.float32)
    interference_level = tf.cast(interference_level, tf.float32)

    # Normalize values with better scaling
    sinr_error = tf.clip_by_value((sinr - sinr_target) / 10.0, -2.0, 2.0)
    snr_norm = tf.clip_by_value(snr / 20.0, -1.0, 1.0)
    interference_norm = tf.clip_by_value((interference_level + 90.0) / 20.0, -1.0, 1.0)

    # Compute reward components with positive baseline
    sinr_reward = 5.0 * (1.0 - tf.abs(sinr_error))  # Max 5.0 when error is 0
    snr_reward = 3.0 * (snr_norm + 1.0) / 2.0       # Range [0, 3.0]
    interference_reward = 2.0 * (1.0 - interference_norm)  # Range [0, 2.0]

    # Softer penalty for below-threshold SINR
    penalty = tf.where(
        sinr < sinr_threshold,
        -2.0 * (sinr_threshold - sinr) / sinr_threshold,  # Gradual penalty
        tf.constant(0.0, dtype=tf.float32)
    )

    # Combine rewards with better baseline
    total_reward = sinr_reward + snr_reward + interference_reward + penalty
    return tf.clip_by_value(total_reward, -5.0, 10.0)  # Ensure reasonable range

def compute_interference(channel_state, beamforming_vectors):
    """
    Compute interference levels for MIMO transmissions
    Args:
        channel_state: Complex channel matrix [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant]
        beamforming_vectors: Beamforming vectors [batch_size, num_rx]
    """
    # Convert to complex tensors
    h = tf.cast(channel_state, tf.complex64)
    w = tf.cast(beamforming_vectors, tf.complex64)
    
    # Get dimensions
    batch_size = tf.shape(h)[0]
    num_rx = tf.shape(h)[1]
    num_rx_ant = tf.shape(h)[2]
    num_tx = tf.shape(h)[3]
    num_tx_ant = tf.shape(h)[4]
    
    # Calculate correct dimensions
    total_tx_elements = num_tx * num_tx_ant
    elements_per_rx = total_tx_elements // num_rx
    
    # Reshape tensors
    h_reshaped = tf.reshape(h, [batch_size, num_rx * num_rx_ant, total_tx_elements])
    w_expanded = tf.expand_dims(w, axis=-1)
    w_tiled = tf.tile(w_expanded, [1, 1, elements_per_rx])
    w_reshaped = tf.reshape(w_tiled, [batch_size, total_tx_elements, 1])
    
    # Initialize interference tensor
    interference = tf.zeros([batch_size], dtype=tf.float32)
    
    # For each receiver
    for i in range(num_rx):
        # Calculate indices for current receiver's antennas
        start_idx = i * num_rx_ant
        end_idx = (i + 1) * num_rx_ant
        
        # Calculate interference for current receiver
        channel_slice = h_reshaped[:, start_idx:end_idx, :]
        desired_signal = tf.abs(tf.matmul(channel_slice, w_reshaped))**2
        total_power = tf.reduce_sum(desired_signal, axis=1)
        interference += total_power - desired_signal[:, 0]
    
    return interference

def compute_sinr(channel_state, beamforming_vectors, noise_power=1.0):
    """
    Compute SINR for MIMO transmissions
    Args:
        channel_state: Complex channel matrix [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant]
        beamforming_vectors: Beamforming vectors [batch_size, num_rx]
        noise_power: Noise power (default: 1.0)
    """
    # Convert to complex tensors
    h = tf.cast(channel_state, tf.complex64)
    w = tf.cast(beamforming_vectors, tf.complex64)
    
    # Get dimensions
    batch_size = tf.shape(h)[0]  # 256
    num_rx = tf.shape(h)[1]      # 4
    num_rx_ant = tf.shape(h)[2]  # 4
    num_tx = tf.shape(h)[3]      # 14
    num_tx_ant = tf.shape(h)[4]  # 64
    
    # Reshape h to [batch_size, num_rx * num_rx_ant, num_tx * num_tx_ant]
    h_reshaped = tf.reshape(h, [batch_size, num_rx * num_rx_ant, num_tx * num_tx_ant])
    
    # Calculate the correct number of elements for w_tiled
    total_tx_elements = num_tx * num_tx_ant
    elements_per_rx = total_tx_elements // num_rx
    
    # Reshape w to match dimensions for matrix multiplication
    w_expanded = tf.expand_dims(w, axis=-1)  # [batch_size, num_rx, 1]
    w_tiled = tf.tile(w_expanded, [1, 1, elements_per_rx])  # [batch_size, num_rx, elements_per_rx]
    w_reshaped = tf.reshape(w_tiled, [batch_size, total_tx_elements, 1])
    
    # Compute desired signal power
    desired_signal = tf.abs(tf.matmul(h_reshaped, w_reshaped))**2
    signal_power = tf.reduce_sum(desired_signal, axis=1)
    
    # Compute interference
    interference = compute_interference(channel_state, beamforming_vectors)
    
    # Compute SINR
    sinr = 10.0 * tf.math.log(signal_power / (interference + noise_power)) / tf.math.log(10.0)
    
    return sinr

def train_sac(training_data, validation_data, config):
    # Convert complex channel data to magnitude and phase
    def preprocess_channel_data(channel_data):
        """
        Preprocess channel data by separating real and imaginary parts
        Args:
            channel_data: Input channel realizations [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant]
        Returns:
            Preprocessed channel data with real and imaginary parts concatenated
        """
        # Convert to complex tensor
        channel_data = tf.cast(channel_data, tf.complex64)
        
        # Separate real and imaginary parts
        real_part = tf.math.real(channel_data)
        imag_part = tf.math.imag(channel_data)
        
        # Stack real and imaginary parts along a new axis
        processed_data = tf.stack([real_part, imag_part], axis=-1)
        
        # Convert to float32 for the neural network
        processed_data = tf.cast(processed_data, tf.float32)
        
        # Ensure the shape matches the expected dimensions
        processed_data.set_shape([None, 4, 4, 14, 64, 2])  # Added dimension for real/imag parts
        
        return processed_data
    
    # Preprocess training and validation data
    training_channels = preprocess_channel_data(training_data[0])
    validation_channels = preprocess_channel_data(validation_data[0])
    
    input_shape = training_channels.shape[1:]  # Updated input shape
    num_actions = MIMO_CONFIG["tx_antennas"]
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
    with tqdm(total=config["episodes"], desc="Training Episodes") as episode_pbar:
        for episode in range(config["episodes"]):
            episode_rewards = []
            
            # Batch training with progress bar
            total_batches = len(training_data[0]) // config["batch_size"]
            with tqdm(total=total_batches, desc=f"Episode {episode+1}", leave=False) as batch_pbar:
                for start in range(0, len(training_data[0]), config["batch_size"]):
                    end = start + config["batch_size"]
                    batch_channels = training_data[0][start:end]
                    batch_snr = training_data[1][start:end]
                    
                    # Convert to tensor and ensure proper shape with batch dimension
                    batch_channels = tf.convert_to_tensor(batch_channels, dtype=tf.float32)
                    
                    # Use gradient tape for tracking gradients
                    with tf.GradientTape(persistent=True) as tape:
                        # Get actions for the entire batch at once
                        actions = sac.get_action(batch_channels)

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

                        # Alpha loss computation
                        alpha_loss = -tf.reduce_mean(sac.alpha * tf.stop_gradient(entropy - 1.0))

                    # Compute gradients
                    critic1_grads = tape.gradient(critic_loss, sac.critic1.trainable_variables)
                    critic2_grads = tape.gradient(critic_loss, sac.critic2.trainable_variables)
                    actor_grads = tape.gradient(actor_loss, sac.actor.trainable_variables)
                    alpha_grads = tape.gradient(alpha_loss, [sac.alpha])

                    # Apply gradients
                    sac.critic1.optimizer.apply_gradients(zip(critic1_grads, sac.critic1.trainable_variables))
                    sac.critic2.optimizer.apply_gradients(zip(critic2_grads, sac.critic2.trainable_variables))
                    sac.actor.optimizer.apply_gradients(zip(actor_grads, sac.actor.trainable_variables))
                    sac.optimizer_alpha.apply_gradients(zip(alpha_grads, [sac.alpha]))

                    # Clean up the tape
                    del tape

                    # Track training history
                    training_history['critic_losses'].append(float(critic_loss))
                    training_history['actor_losses'].append(float(actor_loss))
                    training_history['episode_rewards'].append(float(tf.reduce_mean(rewards)))
                    episode_rewards.append(float(tf.reduce_mean(rewards)))

                    # Update batch progress bar
                    batch_pbar.set_postfix({
                        'critic_loss': f'{float(critic_loss):.3f}',
                        'actor_loss': f'{float(actor_loss):.3f}',
                        'reward': f'{float(tf.reduce_mean(rewards)):.3f}'
                    })
                    batch_pbar.update(1)

            # Update episode progress bar
            avg_episode_reward = np.mean(episode_rewards)
            episode_pbar.set_postfix({
                'avg_reward': f'{avg_episode_reward:.3f}',
                'alpha': f'{float(sac.alpha):.3f}'
            })
            episode_pbar.update(1)

            # Validate performance
            if episode % config["validation_interval"] == 0:
                val_reward = validate_model(sac, validation_data)
                tqdm.write(f"\nValidation reward at episode {episode+1}: {val_reward:.3f}\n")

            # Create directories for saving results
            results_dir = "./results"
            metrics_dir = os.path.join(results_dir, "performance_metrics")
            viz_dir = os.path.join(results_dir, "visualizations")
            xai_dir = os.path.join(results_dir, "xai_analysis")
            
            for directory in [metrics_dir, viz_dir, xai_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            # Save performance metrics
            metrics = {
                'episode_rewards': training_history['episode_rewards'],
                'critic_losses': training_history['critic_losses'],
                'actor_losses': training_history['actor_losses']
            }
            np.save(f"{metrics_dir}/training_metrics.npy", metrics)
            
            # Save training plots
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['episode_rewards'])
            plt.title('Episode Rewards')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.savefig(f"{viz_dir}/rewards_plot.png")
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['critic_losses'])
            plt.title('Critic Losses')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.savefig(f"{viz_dir}/critic_losses_plot.png")
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['actor_losses'])
            plt.title('Actor Losses')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.savefig(f"{viz_dir}/actor_losses_plot.png")
            plt.close()

            # Save models
            model_dir = os.path.join(results_dir, "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            sac.actor.save(os.path.join(model_dir, "actor_model"))
            sac.critic1.save(os.path.join(model_dir, "critic1_model"))
            sac.critic2.save(os.path.join(model_dir, "critic2_model"))
            np.save(os.path.join(model_dir, "alpha.npy"), sac.alpha.numpy())

            print("Training complete.")
            print(f"\nResults saved in:")
            print(f"Performance metrics: {metrics_dir}")
            print(f"Visualizations: {viz_dir}")
            print(f"Models: {model_dir}")
            
            return sac, training_history

# Validation function
def validate_model(sac, validation_data):
    val_channels, val_snr = validation_data
    total_reward = 0
    
    total_batches = len(val_channels) // CONFIG["batch_size"]
    with tqdm(total=total_batches, desc="Validating", leave=False) as val_pbar:
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
            
            batch_reward = np.mean(rewards)
            total_reward += np.sum(rewards)
            
            val_pbar.set_postfix({'batch_reward': f'{batch_reward:.3f}'})
            val_pbar.update(1)

        avg_reward = total_reward / len(validation_data[0])
        return avg_reward

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
