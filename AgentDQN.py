from singleLinEnv_DQN import DQN_Env
import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle

# --- Reproducibility ---
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)

# --- Parameters ---
episodes = 500
max_step = 20
target_update_freq = 10
gamma = 0.95
learning_rate = 0.001
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 10000
train_freq = 4

# --- Save/load paths ---
save_dir = r"C:\Users\abdel\OneDrive\Desktop\Wireless Systems\DLRL-Power-Ctrl\simulationImages\DQN"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "dqn_model.h5")
epsilon_path = os.path.join(save_dir, "epsilon.pkl")
rewards_path = os.path.join(save_dir, "rewards_history.npy")


class DQNagent:
    def __init__(self, state_dim, action_dim, lr=learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.train_freq = train_freq
        self.step_count = 0

        # Build networks
        self.policy_net = self.build_model()
        self.target_net = self.build_model()
        self.update_target_network()

        # Load model if exists
        if os.path.exists(model_path):
            print("Loading saved model...")
            self.policy_net = models.load_model(model_path)
            self.target_net = models.load_model(model_path)
            # Load epsilon if saved
            if os.path.exists(epsilon_path):
                with open(epsilon_path, "rb") as f:
                    self.epsilon = pickle.load(f)
            print(f"Continuing training with epsilon={self.epsilon:.3f}")

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_dim, activation="linear")
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss="mse")
        return model

    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def action_selection(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.policy_net.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.step_count += 1
        if len(self.memory) >= self.batch_size and self.step_count % self.train_freq == 0:
            self.train()

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        q_current = self.policy_net.predict(states, verbose=0)
        q_next = self.target_net.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next[i])
            q_current[i, actions[i]] = target

        self.policy_net.fit(states, q_current, epochs=1, verbose=0)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# --- Training Loop ---
env = DQN_Env()
agent = DQNagent(env.observation_space.shape[0], env.action_space.n, lr=learning_rate)
rewards_history = []

# Load previous rewards if available
if os.path.exists(rewards_path):
    rewards_history = list(np.load(rewards_path).tolist())
    print(f"Loaded previous rewards, length={len(rewards_history)}")

moving_avg_history = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_step):
        action = agent.action_selection(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done: break

    agent.decay_epsilon()
    if episode % target_update_freq == 0:
        agent.update_target_network()

    rewards_history.append(total_reward)

    # 10-episode moving average
    if len(rewards_history) >= 10:
        moving_avg_history.append(np.mean(rewards_history[-10:]))

    if episode % 50 == 0:
        print(f"Episode {episode:3d} | Reward: {total_reward:.2f} | Avg (last 10): {np.mean(rewards_history[-10:]):.2f} | Eps: {agent.epsilon:.3f}")

# --- Save everything ---
agent.policy_net.save(model_path)
with open(epsilon_path, "wb") as f:
    pickle.dump(agent.epsilon, f)
np.save(rewards_path, np.array(rewards_history))
print(f"Model, epsilon, and rewards saved in: {save_dir}")

# --- Plot ---
plt.plot(rewards_history, alpha=0.3, label="Raw Reward")
plt.plot(range(9, len(rewards_history)), moving_avg_history, label="10-Ep Moving Avg", color='red')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plot_path = os.path.join(save_dir, "DQN_training_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Plot saved at: {plot_path}")
