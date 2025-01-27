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
        # Add shape validation
        if len(self.buffer) > 0:
            if (state.shape != self.buffer[0][0].shape or 
                action.shape != self.buffer[0][1].shape or
                not np.isscalar(reward) or  # Ensure reward is a scalar
                next_state.shape != self.buffer[0][3].shape):
                raise ValueError(f"Shape mismatch in buffer push: \n"
                            f"state: {state.shape}, expected: {self.buffer[0][0].shape}\n"
                            f"action: {action.shape}, expected: {self.buffer[0][1].shape}\n"
                            f"next_state: {next_state.shape}, expected: {self.buffer[0][3].shape}")
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Stack arrays and ensure rewards are reshaped properly
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states)
        )

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
        self.target_entropy = -np.log(1.0/num_actions) * 0.98
        
        # Main networks
        self.actor = self.build_actor(input_shape, num_actions, learning_rates["actor"])
        self.critic1 = self.build_critic(input_shape, num_actions, learning_rates["critic"])
        self.critic2 = self.build_critic(input_shape, num_actions, learning_rates["critic"])
        
        # Target networks
        self.target_critic1 = self.build_critic(input_shape, num_actions, learning_rates["critic"])
        self.target_critic2 = self.build_critic(input_shape, num_actions, learning_rates["critic"])
        
        # Copy weights to target networks
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        
        # Temperature parameter
        self.log_alpha = tf.Variable(np.log(learning_rates["alpha"]), dtype=tf.float32)
        self.alpha = tf.exp(self.log_alpha)
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rates["actor"],
            clipnorm=1.0
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rates["critic"],
            clipnorm=1.0
        )
        self.alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rates["alpha"],
            clipnorm=1.0
        )

    def build_actor(self, input_shape, num_actions, lr):
        """Enhanced actor network with regularization and residual connections"""
        inputs = layers.Input(shape=(np.prod(input_shape),))
        
        # Initial normalization
        x = layers.BatchNormalization()(inputs)
        
        # First dense block with residual connection
        residual = x
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        if residual.shape[-1] == x.shape[-1]:
            x = layers.Add()([x, residual])
        
        # Second dense block with residual connection
        residual = x
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512)(x)  # Match dimensions for residual
        if residual.shape[-1] == x.shape[-1]:
            x = layers.Add()([x, residual])
        
        # Third dense block
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Output layer with scaled initialization and noise
        outputs = layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
        )(x)
        
        # Add noise for exploration
        noise = layers.GaussianNoise(0.1)(outputs)
        outputs = layers.Activation('softmax')(outputs + noise)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def build_critic(self, input_shape, num_actions, lr):
        """Enhanced critic network with separate state and action processing"""
        # State input branch
        state_input = layers.Input(shape=(np.prod(input_shape),))
        state_x = layers.BatchNormalization()(state_input)
        state_x = layers.Dense(512, activation='relu')(state_x)
        state_x = layers.BatchNormalization()(state_x)
        state_x = layers.Dropout(0.1)(state_x)
        
        # Action input branch
        action_input = layers.Input(shape=(num_actions,))
        action_x = layers.Dense(512, activation='relu')(action_input)
        action_x = layers.BatchNormalization()(action_x)
        
        # Combine state and action pathways
        combined = layers.Concatenate()([state_x, action_x])
        
        # Process combined features
        x = layers.Dense(512, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output Q-value
        q_value = layers.Dense(1, 
                            kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
                            )(x)
        
        model = tf.keras.Model([state_input, action_input], q_value)
        return model

    def update_target_networks(self, tau=0.005):
        """Soft update target networks"""
        for target, source in [(self.target_critic1, self.critic1), 
                            (self.target_critic2, self.critic2)]:
            for target_weight, source_weight in zip(target.weights, source.weights):
                target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def get_action(self, state, training=True):
        """Get actions with optional noise for exploration"""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
            
        action_probs = self.actor(state, training=training)
        
        if training:
            # Add exploration noise during training
            noise = tf.random.normal(shape=action_probs.shape, mean=0.0, stddev=0.1)
            action_probs = tf.nn.softmax(tf.math.log(action_probs + 1e-10) + noise)
        
        actions = tf.random.categorical(tf.math.log(action_probs + 1e-10), 1)
        return tf.one_hot(tf.squeeze(actions), depth=self.num_actions)

    def save_models(self, path):
        """Save all models and parameters"""
        self.actor.save(f"{path}/actor")
        self.critic1.save(f"{path}/critic1")
        self.critic2.save(f"{path}/critic2")
        self.target_critic1.save(f"{path}/target_critic1")
        self.target_critic2.save(f"{path}/target_critic2")
        np.save(f"{path}/alpha.npy", self.alpha.numpy())

    def load_models(self, path):
        """Load all models and parameters"""
        self.actor = tf.keras.models.load_model(f"{path}/actor")
        self.critic1 = tf.keras.models.load_model(f"{path}/critic1")
        self.critic2 = tf.keras.models.load_model(f"{path}/critic2")
        self.target_critic1 = tf.keras.models.load_model(f"{path}/target_critic1")
        self.target_critic2 = tf.keras.models.load_model(f"{path}/target_critic2")
        self.alpha.assign(np.load(f"{path}/alpha.npy"))

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
    # Cast inputs to float32
    snr = tf.cast(snr, tf.float32)
    sinr = tf.cast(sinr, tf.float32)
    interference_level = tf.cast(interference_level, tf.float32)
    
    # Constants
    sinr_target = tf.constant(20.0, dtype=tf.float32)
    sinr_threshold = tf.constant(10.0, dtype=tf.float32)
    
    # SINR reward component (weighted more heavily)
    sinr_error = (sinr - sinr_target) / sinr_target
    sinr_reward = 3.0 * tf.exp(-tf.abs(sinr_error))  # Exponential reward
    
    # SNR reward component
    snr_norm = snr / 40.0  # Normalize SNR
    snr_reward = 2.0 * tf.sigmoid(snr_norm)  # Bounded positive reward
    
    # Interference penalty
    interference_norm = interference_level / 100.0
    interference_penalty = -1.0 * tf.sigmoid(interference_norm)
    
    # Additional reward for exceeding threshold
    threshold_bonus = tf.where(
        sinr > sinr_threshold,
        1.0,
        0.0
    )
    
    # Combine rewards with proper scaling
    total_reward = sinr_reward + snr_reward + interference_penalty + threshold_bonus
    
    # Wider range for rewards
    return tf.clip_by_value(total_reward, -5.0, 5.0)

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
    Train the Soft Actor-Critic model with logging and notifications.
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
        Validate shapes of inputs.
        """
        if len(batch_channels.shape) != 2:
            raise ValueError(f"Expected batch_channels shape [batch_size, flattened_features], got {batch_channels.shape}")
        if len(actions.shape) != 2:
            raise ValueError(f"Expected actions shape [batch_size, num_actions], got {actions.shape}")

    # Initialize SAC model
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
        staircase=True
    )

    # Update optimizers to use lr_schedule
    sac.actor_optimizer.learning_rate = lr_schedule
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

    # In train_sac function, modify the warmup section:
    print("Warming up replay buffer...")
    warmup_episodes = 10
    for _ in range(warmup_episodes):
        # Get a batch of states from training data
        start_idx = _ * config["batch_size"]
        end_idx = start_idx + config["batch_size"]
        
        # Get batch data
        batch_channels_orig = training_channels_orig[start_idx:end_idx]
        batch_channels_flat = training_channels_flat[start_idx:end_idx]
        batch_snr = training_snr[start_idx:end_idx]
        
        # Generate random actions with correct shape
        actions = tf.random.uniform((config["batch_size"], MIMO_CONFIG["tx_antennas"]))
        actions = tf.nn.softmax(actions, axis=-1)  # Normalize actions
        
        # Compute rewards
        sinr_values = compute_sinr(batch_channels_orig, actions)
        interference_levels = compute_interference(batch_channels_orig, actions)
        rewards = compute_reward(batch_snr, sinr_values, interference_levels)
        
        # Convert tensors to numpy arrays
        rewards_np = rewards.numpy()
        batch_channels_flat_np = batch_channels_flat.numpy()
        actions_np = actions.numpy()
        
        # Process each sample individually with correct shapes
        for i in range(config["batch_size"]):
            s = batch_channels_flat_np[i]  # Single state
            a = actions_np[i]              # Single action
            r = float(np.mean(rewards_np[i]))  # Take mean if reward is multi-dimensional
            ns = batch_channels_flat_np[i] # Single next state
            
            # Push to replay buffer
            replay_buffer.push(s, a, r, ns)

    # Training loop
    with tqdm(total=config["episodes"], desc="Training Episodes") as episode_pbar:
        for episode in range(config["episodes"]):
            episode_rewards = []
            
            # Batch training with progress bar
            total_batches = len(training_data[0]) // config["batch_size"]
            with tqdm(total=total_batches, desc=f"Episode {episode+1}", leave=False) as batch_pbar:
                for start in range(0, len(training_data[0]), config["batch_size"]):
                    end = start + config["batch_size"]
                    batch_channels_orig = training_channels_orig[start:end]  # Original structure for SINR
                    batch_channels_flat = training_channels_flat[start:end]  # Flattened for NN
                    batch_snr = training_data[1][start:end]
                    
                    with tf.GradientTape(persistent=True) as tape:
                        # Get current actions and their probabilities
                        current_actions = sac.get_action(batch_channels_flat)
                        current_action_probs = sac.actor(batch_channels_flat)
                        current_log_probs = tf.math.log(current_action_probs + 1e-10)
                        
                        # Validate shapes
                        validate_shapes(batch_channels_flat, current_actions)
                        
                        # Compute immediate rewards using original structure
                        sinr_values = compute_sinr(batch_channels_orig, current_actions)
                        interference_levels = compute_interference(batch_channels_orig, current_actions)
                        
                        # Compute rewards for each sample individually
                        rewards = []
                        for snr, sinr, interference in zip(batch_snr, sinr_values, interference_levels):
                            reward = compute_reward(snr, sinr, interference)
                            if isinstance(reward, tf.Tensor):
                                reward = tf.reduce_mean(reward)
                            rewards.append(float(reward))
                        
                        # Convert rewards to tensor
                        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                        
                        # Get next state actions and probabilities for target computation
                        next_actions = sac.get_action(batch_channels_flat)
                        next_action_probs = sac.actor(batch_channels_flat)
                        next_log_probs = tf.math.log(next_action_probs + 1e-10)
                        
                        # Compute current Q-values
                        current_q1 = sac.critic1([batch_channels_flat, current_actions])
                        current_q2 = sac.critic2([batch_channels_flat, current_actions])
                        
                        # Compute target Q-values using target networks
                        target_q1 = sac.target_critic1([batch_channels_flat, next_actions])
                        target_q2 = sac.target_critic2([batch_channels_flat, next_actions])
                        
                        # Use minimum Q-value for targets (Double Q-learning)
                        min_target_q = tf.minimum(target_q1, target_q2)
                        
                        # Compute entropy term
                        entropy = -tf.reduce_sum(current_action_probs * current_log_probs, axis=-1, keepdims=True)
                        
                        # Compute target value with entropy regularization
                        target_value = rewards[:, tf.newaxis] + config["gamma"] * (min_target_q - sac.alpha * next_log_probs)
                        
                        # Compute critic losses
                        critic1_loss = tf.reduce_mean(tf.square(target_value - current_q1))
                        critic2_loss = tf.reduce_mean(tf.square(target_value - current_q2))
                        critic_loss = critic1_loss + critic2_loss
                        
                        # Compute actor loss with entropy regularization
                        q_values = tf.minimum(current_q1, current_q2)
                        actor_loss = tf.reduce_mean(sac.alpha * current_log_probs - q_values)
                        
                        # Compute temperature (alpha) loss
                        alpha_loss = -tf.reduce_mean(
                            sac.alpha * tf.stop_gradient(entropy + config["target_entropy"])
                        )

                        # Store experience in replay buffer
                        batch_channels_flat_np = batch_channels_flat.numpy()
                        current_actions_np = current_actions.numpy()
                        rewards_np = rewards.numpy()
                        
                        for i in range(config["batch_size"]):
                            replay_buffer.push(
                                batch_channels_flat_np[i],
                                current_actions_np[i],
                                float(rewards_np[i]),
                                batch_channels_flat_np[i]
                            )

                        # Compute and clip gradients
                        critic1_grads = tape.gradient(critic_loss, sac.critic1.trainable_variables)
                        critic2_grads = tape.gradient(critic_loss, sac.critic2.trainable_variables)
                        actor_grads = tape.gradient(actor_loss, sac.actor.trainable_variables)
                        alpha_grads = tape.gradient(alpha_loss, [sac.alpha])

                        # Clip gradients
                        critic1_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in critic1_grads]
                        critic2_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in critic2_grads]
                        actor_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in actor_grads]

                        # Apply gradients
                        sac.critic1.optimizer.apply_gradients(zip(critic1_grads, sac.critic1.trainable_variables))
                        sac.critic2.optimizer.apply_gradients(zip(critic2_grads, sac.critic2.trainable_variables))
                        sac.actor.optimizer.apply_gradients(zip(actor_grads, sac.actor.trainable_variables))
                        sac.optimizer_alpha.apply_gradients(zip(alpha_grads, [sac.alpha]))

                    # Update target networks
                    sac.update_target_networks(config["tau"])

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
                val_reward = validate_model(sac, ((validation_channels_orig, validation_channels_flat), validation_snr))
                training_history['validation_rewards'].append(float(val_reward))
                training_history['validation_episodes'].append(episode)
                
                # Add this line to log training progress
                _log_training_progress(
                    episode=episode,
                    avg_reward=avg_episode_reward,
                    critic_loss=float(critic_loss),
                    actor_loss=float(actor_loss),
                    val_reward=val_reward
                )
                
                tqdm.write(f"\nValidation reward at episode {episode+1}: {val_reward:.3f}\n")

            # Save results and visualizations
            if episode % config["save_interval"] == 0:
                _save_results(sac, training_history, episode)

    print("Training complete.")
    print(f"\nResults saved in:")
    print(f"Performance metrics: ./results/performance_metrics")
    print(f"Visualizations: ./results/visualizations")
    print(f"Models: ./results/models")
    
    return sac, training_history

def _log_training_progress(episode, avg_reward, critic_loss, actor_loss, val_reward):
    """
    Log training progress and key metrics.
    """
    log_message = (
        f"Episode {episode + 1}:\n"
        f"  Average Reward: {avg_reward:.3f}\n"
        f"  Critic Loss: {critic_loss:.3f}\n"
        f"  Actor Loss: {actor_loss:.3f}\n"
        f"  Validation Reward: {val_reward:.3f}\n"
    )
    print(log_message)

def _save_results(sac, history, episode):
    """
    Save training results, models, and visualizations.
    """
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
        'episode_rewards': history['episode_rewards'],
        'critic_losses': history['critic_losses'],
        'actor_losses': history['actor_losses']
    }
    np.save(f"{metrics_dir}/training_metrics.npy", metrics)
    
    # Save training plots
    plt.figure(figsize=(10, 6))
    plt.plot(history['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig(f"{viz_dir}/rewards_plot.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['critic_losses'])
    plt.title('Critic Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(f"{viz_dir}/critic_losses_plot.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['actor_losses'])
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
            "batch_size": 128,  # Smaller batch size for better stability
            "actor_lr": 1e-4,   # Slightly lower learning rate
            "critic_lr": 1e-4,
            "alpha_lr": 1e-4,
            "validation_interval": 5,
            "save_interval": 10,
            "warmup_episodes": 20,  # More warmup episodes
            "target_entropy": -np.log(1.0/MIMO_CONFIG["tx_antennas"]) * 0.98,  # Target entropy for alpha adaptation
            "gamma": 0.99,  # Discount factor
            "tau": 0.005,  # Soft update coefficient
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
