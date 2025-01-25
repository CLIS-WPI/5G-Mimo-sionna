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
import gc
import json
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
import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)
    
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    try:
        data = np.load(file_path, allow_pickle=True).item()
        if "channel_realizations" not in data or "snr" not in data:
            raise ValueError("Dataset missing required fields")
        return data["channel_realizations"], data["snr"]
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {str(e)}")
        raise
def preprocess_channel_data(channel_data):
    """
    Preprocess channel data by separating real and imaginary parts while maintaining structure
    Args:
        channel_data: Input channel realizations [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant]
    Returns:
        Original channel data for SINR computation and flattened data for the neural network
    """
    # Convert to complex tensor
    channel_data = tf.cast(channel_data, tf.complex64)
        
    # Keep original data for SINR computation
    original_data = channel_data
        
    # Separate real and imaginary parts
    real_part = tf.math.real(channel_data)
    imag_part = tf.math.imag(channel_data)
        
    # Stack real and imaginary parts along a new axis
    processed_data = tf.stack([real_part, imag_part], axis=-1)
        
    # Convert to float32 for the neural network
    processed_data = tf.cast(processed_data, tf.float32)
        
    # Reshape to combine all features
    batch_size = tf.shape(processed_data)[0]
    feature_dim = np.prod(processed_data.shape[1:])
    flattened_data = tf.reshape(processed_data, [batch_size, -1])
        
    return original_data, flattened_data

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
        """
        Build actor network with correct shape handling
        Args:
            input_shape: Shape of flattened input tensor
            num_actions: Number of possible actions
            lr: Learning rate
        """
        # Input layer
        inputs = layers.Input(shape=(np.prod(input_shape),))  # Flattened input
        
        # Dense layers
        x = layers.Dense(1024, activation="relu")(inputs)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        
        # Output layer
        outputs = layers.Dense(num_actions, activation="softmax")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        
        return model

    def build_critic(self, input_shape, num_actions, lr):
        # State input branch should match flattened input
        state_input = layers.Input(shape=(np.prod(input_shape),))  # Flattened input
        x = layers.Dense(256, activation="relu")(state_input)
        x = layers.Dense(128, activation="relu")(x)
        
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
    # Cast all inputs to float32
    snr = tf.cast(snr, tf.float32)
    sinr = tf.cast(sinr, tf.float32)
    interference_level = tf.cast(interference_level, tf.float32)
    
    # Adjust target and threshold values
    sinr_target = tf.constant(20.0, dtype=tf.float32)
    sinr_threshold = tf.constant(5.0, dtype=tf.float32)  # Reduced from 10.0

    # Softer normalization with reduced penalties
    sinr_error = tf.clip_by_value((sinr - sinr_target) / 30.0, -1.0, 1.0)
    snr_norm = tf.clip_by_value(snr / 40.0, -1.0, 1.0)
    interference_norm = tf.clip_by_value(interference_level / 50.0, -1.0, 1.0)

    # Adjusted reward weights
    sinr_reward = 2.0 * (1.0 - tf.abs(sinr_error))
    snr_reward = 1.0 * snr_norm
    interference_reward = 1.0 * (1.0 - interference_norm)

    # Softer penalty
    penalty = tf.where(
        sinr < sinr_threshold,
        -0.5 * (sinr_threshold - sinr) / sinr_threshold,
        tf.constant(0.0, dtype=tf.float32)
    )

    total_reward = sinr_reward + snr_reward + interference_reward + penalty
    return tf.clip_by_value(total_reward, -2.0, 2.0)  # Reduced range

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
    """
    Train the Soft Actor-Critic model
    """
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)

    # Process training and validation data
    (training_channels_orig, training_channels_flat), training_snr = training_data 
    (validation_channels_orig, validation_channels_flat), validation_snr = validation_data

    # Calculate input shape after preprocessing
    input_shape = training_channels_flat.shape[1:]

    def validate_shapes(batch_channels, actions):
        """
        Validate shapes of inputs
        """
        if len(batch_channels.shape) != 2:
            raise ValueError(f"Expected batch_channels shape [batch_size, flattened_features], got {batch_channels.shape}")
        if len(actions.shape) != 2:
            raise ValueError(f"Expected actions shape [batch_size, num_actions], got {actions.shape}")

    # Initialize SAC model (only once)
    sac = SoftActorCritic(
        input_shape=input_shape,
        num_actions=MIMO_CONFIG["tx_antennas"],
        learning_rates={
            "actor": config["actor_lr"],
            "critic": config["critic_lr"],
            "alpha": config["alpha_lr"]
        }
    )

    # Add learning rate decay (only once)
    initial_lr = config["actor_lr"]
    decay_steps = config["episodes"] // 2
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True)

    # Update optimizers to use lr_schedule
    sac.actor.optimizer.learning_rate = lr_schedule
    sac.critic1.optimizer.learning_rate = lr_schedule
    sac.critic2.optimizer.learning_rate = lr_schedule

    # Initialize training history
    training_history = {
        'episode_rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'validation_rewards': [],
        'validation_episodes': []
    }

    # Warmup replay buffer
    print("Warming up replay buffer...")
    warmup_episodes = 10
    for _ in range(warmup_episodes):
        states = training_channels_flat[:config["batch_size"]]
        actions = tf.random.uniform((config["batch_size"], sac.num_actions))
        next_states = states  # In this case, next state is same as current
        rewards = compute_reward(
            training_snr[:config["batch_size"]], 
            compute_sinr(training_channels_orig[:config["batch_size"]], actions),
            compute_interference(training_channels_orig[:config["batch_size"]], actions)
        )
        for s, a, r, ns in zip(states, actions, rewards, next_states):
            replay_buffer.push(s, a, r, ns)

    # Initialize training history
    training_history = {
        'episode_rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'validation_rewards': [],
        'validation_episodes': []
    }


    # Inside train_sac function, add checkpoint saving:
    def save_checkpoint(sac, history, episode, save_dir):
        checkpoint_dir = os.path.join(save_dir, f"checkpoint_ep_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        sac.actor.save(os.path.join(checkpoint_dir, "actor"))
        sac.critic1.save(os.path.join(checkpoint_dir, "critic1"))
        sac.critic2.save(os.path.join(checkpoint_dir, "critic2"))
        
        # Save alpha
        np.save(os.path.join(checkpoint_dir, "alpha.npy"), sac.alpha.numpy())
        
        # Save history
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
            
    
    # Calculate input shape after preprocessing
    input_shape = training_channels_flat.shape[1:]
    def validate_shapes(batch_channels, actions):
        """
        Validate shapes of inputs
        Args:
            batch_channels: Preprocessed channel data
            actions: Action vectors from the actor network
        """
        if len(batch_channels.shape) != 2:
            raise ValueError(f"Expected batch_channels shape [batch_size, flattened_features], got {batch_channels.shape}")
        if len(actions.shape) != 2:
            raise ValueError(f"Expected actions shape [batch_size, num_actions], got {actions.shape}")
    
    sac = SoftActorCritic(
        input_shape=input_shape,
        num_actions=MIMO_CONFIG["tx_antennas"],
        learning_rates={
            "actor": config["actor_lr"],
            "critic": config["critic_lr"],
            "alpha": config["alpha_lr"]
        }
    )

    # Add learning rate decay
    initial_lr = config["actor_lr"]
    decay_steps = config["episodes"] // 2
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True)

    # Update optimizers to use lr_schedule
    sac.actor.optimizer.learning_rate = lr_schedule
    sac.critic1.optimizer.learning_rate = lr_schedule
    sac.critic2.optimizer.learning_rate = lr_schedule

    # Training loop
    with tqdm(total=config["episodes"], desc="Training Episodes") as episode_pbar:
        for episode in range(config["episodes"]):
            episode_rewards = []
            
            # Batch training with progress bar
            total_batches = len(training_data[0]) // config["batch_size"]
            with tqdm(total=total_batches, desc=f"Episode {episode+1}", leave=False) as batch_pbar:
                # In the training loop, modify the batch processing:
                for start in range(0, len(training_data[0]), config["batch_size"]):
                    end = start + config["batch_size"]
                    batch_channels_orig = training_channels_orig[start:end]  # Original structure for SINR
                    batch_channels_flat = training_channels_flat[start:end]  # Flattened for NN
                    batch_snr = training_data[1][start:end]
                    
                    with tf.GradientTape(persistent=True) as tape:
                        # Get actions using flattened data
                        actions = sac.get_action(batch_channels_flat)

                        # Validate shapes before proceeding
                        validate_shapes(batch_channels_flat, actions)
                        
                        # Compute rewards using original structure
                        sinr_values = compute_sinr(batch_channels_orig, actions)
                        interference_levels = compute_interference(batch_channels_orig, actions)
                        rewards = tf.convert_to_tensor(
                            [compute_reward(snr, sinr, interference) 
                            for snr, sinr, interference in zip(batch_snr, sinr_values, interference_levels)],
                            dtype=tf.float32
                        )
                        
                        replay_buffer.push(batch_channels_flat, actions, rewards, batch_channels_flat)  # Note: in this case next_state is same as current state
                        
                        # Sample from replay buffer for training
                        if len(replay_buffer) > config["batch_size"]:
                            states, actions, rewards, next_states = replay_buffer.sample(config["batch_size"])

                        # Critic loss computation
                        critic1_value = sac.critic1([batch_channels_flat, actions])
                        critic2_value = sac.critic2([batch_channels_flat, actions])
                        
                        # Use minimum of critics (Double Q-learning)
                        min_critic = tf.minimum(critic1_value, critic2_value)
                        critic_loss = tf.reduce_mean(tf.square(rewards - tf.squeeze(min_critic)))
                        
                        # Actor loss with entropy regularization
                        actor_probs = sac.actor(batch_channels_flat)
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

                        # Add gradient clipping
                        critic1_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in critic1_grads]
                        critic2_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in critic2_grads]
                        actor_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in actor_grads]

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
            # In the training loop:
            if episode % config["validation_interval"] == 0:
                val_reward = validate_model(sac, ((validation_channels_orig, validation_channels_flat), validation_snr))
                training_history['validation_rewards'].append(float(val_reward))
                training_history['validation_episodes'].append(episode)
                tqdm.write(f"\nValidation reward at episode {episode+1}: {val_reward:.3f}\n")
            # Save results and visualizations
            if episode % config["save_interval"] == 0:
                # Create directories for saving results
                results_dir = "./results"
                metrics_dir = os.path.join(results_dir, "performance_metrics")
                viz_dir = os.path.join(results_dir, "visualizations")
                model_dir = os.path.join(results_dir, "models")
                
                for directory in [metrics_dir, viz_dir, model_dir]:
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
    """
    Validate model performance
    Args:
        sac: Trained SAC model
        validation_data: Tuple of ((original_channels, flattened_channels), snr)
    Returns:
        Average reward across validation set
    """
    # Unpack validation data correctly
    (val_channels_orig, val_channels_flat), val_snr = validation_data
    total_reward = 0
    
    total_batches = len(val_channels_orig) // CONFIG["batch_size"]
    with tqdm(total=total_batches, desc="Validating", leave=False) as val_pbar:
        for start in range(0, len(val_channels_orig), CONFIG["batch_size"]):
            end = start + CONFIG["batch_size"]
            # Get batch data
            batch_channels_orig = val_channels_orig[start:end]
            batch_channels_flat = val_channels_flat[start:end]
            batch_snr = val_snr[start:end]

            # Get actions using flattened data
            actions = sac.get_action(batch_channels_flat)
            
            # Compute metrics using original structure
            sinr_values = compute_sinr(batch_channels_orig, actions)
            interference_levels = compute_interference(batch_channels_orig, actions)
            
            # Calculate rewards using the same reward function
            rewards = np.array([compute_reward(snr, sinr, interference) 
                            for snr, sinr, interference in zip(batch_snr, sinr_values, interference_levels)])
            
            batch_reward = np.mean(rewards)
            total_reward += np.sum(rewards)
            
            val_pbar.set_postfix({'batch_reward': f'{batch_reward:.3f}'})
            val_pbar.update(1)

        avg_reward = total_reward / len(val_channels_orig)
        return avg_reward

if __name__ == "__main__":
    try:
        # Load datasets
        print("Loading datasets...")
        training_data = load_dataset(OUTPUT_FILES["training_data"])
        validation_data = load_dataset(OUTPUT_FILES["validation_data"])

        # Define training configuration
        train_config = {
            "episodes": CONFIG["number_of_episodes"],
            "batch_size": CONFIG["mini_batch_size"],
            "actor_lr": CONFIG["actor_lr"],
            "critic_lr": CONFIG["critic_lr"],
            "alpha_lr": CONFIG["alpha_lr"],
            "validation_interval": CONFIG["validation_interval"],
            "save_interval": CONFIG.get("checkpoint_interval", 10)
        }

        # Process data
        training_channels_orig, training_channels_flat = preprocess_channel_data(training_data[0])
        validation_channels_orig, validation_channels_flat = preprocess_channel_data(validation_data[0])

        # Create processed datasets
        processed_training_data = ((training_channels_orig, training_channels_flat), training_data[1])
        processed_validation_data = ((validation_channels_orig, validation_channels_flat), validation_data[1])

        # Train the model
        sac, history = train_sac(processed_training_data, processed_validation_data, train_config)
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
