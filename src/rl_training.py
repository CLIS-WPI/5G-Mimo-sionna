# src/rl_training.py
# This script implements the training of a Soft Actor-Critic (SAC) model for optimizing beamforming in MIMO systems 
# based on simulated channel realizations. The model aims to dynamically adjust the transmit beamforming vector 
# to improve system performance based on varying Signal-to-Noise Ratios (SNR) and channel conditions. 

# Inputs:
# - Training and validation datasets, which include channel realizations and corresponding SNR values.
#   The datasets are loaded from pre-generated .npy files containing the data produced by the dataset generator script.
#   The data consists of:
#   - channel_realizations: A tensor representing the generated channel realizations (e.g., Rayleigh fading).
#   - snr: A vector of corresponding Signal-to-Noise Ratio (SNR) values used as rewards for training.
#
# Outputs:
# - The SAC model's parameters are updated after each episode through gradient-based optimization.
# - The actor and critic networks (and their weights) are refined iteratively to improve the beamforming policy.
# - Training progress and validation rewards are printed during training to monitor the model's performance.
#
# The model uses a policy-based approach to maximize long-term rewards by improving the beamforming decisions,
# leveraging a reinforcement learning paradigm, with the reward being the SNR for simplicity in this case.

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
        inputs = layers.Input(shape=input_shape)
        actions = layers.Input(shape=(num_actions,))
        x = layers.Flatten()(inputs)
        x = layers.Concatenate()([x, actions])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model([inputs, actions], outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def get_action(self, state):
        # Ensure state is in the correct batch format (batch_size, state_shape)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)  # Add batch dimension if it's missing
        action_prob = self.actor(state)
        action = tf.random.categorical(action_prob, 1)
        return action.numpy().flatten()


# Training function
def train_sac(training_data, validation_data, config):
    input_shape = training_data[0].shape[1:]  # Exclude batch size
    num_actions = MIMO_CONFIG["tx_antennas"]  # Beamforming actions correspond to TX antennas

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

            # Convert complex values to real parts (remove imaginary part)
            batch_channels = tf.math.real(batch_channels)

            # Simulate actions and compute rewards (dummy example for now)
            # Ensure actions are generated for the correct batch size
            actions = np.array([sac.get_action(state) for state in batch_channels])

            # Ensure actions have the correct shape (batch_size, num_actions)
            actions = np.reshape(actions, [-1, num_actions])  # Flatten actions to the correct shape

            # Ensure batch size is the same for batch_channels and actions
            if batch_channels.shape[0] != actions.shape[0]:
                raise ValueError(f"Batch size mismatch: batch_channels size {batch_channels.shape[0]}, actions size {actions.shape[0]}")

            rewards = batch_snr  # Assume SNR as reward for simplicity

            # Compute critic losses and actor updates
            with tf.GradientTape(persistent=True) as tape:
                # Ensure batch_channels has correct dimensions for critic
                batch_channels = tf.expand_dims(batch_channels, axis=-1)  # Add the necessary dimension

                # Concatenate the batch_channels (state) with the actions
                inputs = [batch_channels, actions]

                # Critic loss computation (TD Error)
                critic1_value = sac.critic1(inputs)
                critic2_value = sac.critic2(inputs)

                # Use the min between the two critic estimates (Double Q-Learning)
                critic_loss = tf.reduce_mean(tf.square(rewards - tf.minimum(critic1_value, critic2_value)))

                # Actor loss using the entropy regularized objective
                actor_prob = sac.actor(batch_channels)
                actor_loss = -tf.reduce_mean(tf.math.log(actor_prob) * rewards)  # Maximize reward

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
        total_reward += np.sum(batch_snr)  # Sum rewards (for validation)

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
